import copy
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics


# we use prediction loss to conduct our attacks
# prediction loss: for a given sample (x, y), every local model will has a prediction loss on it. we consider the party who has the smallest prediction loss owns the sample.


def calculate_cv(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = (std_dev / mean) # * 100
    return cv

def _safe_prob(probs, small_value=1e-30):
    return np.maximum(probs, small_value)

def uncertainty(probability, n_classes):
    uncert = []
    for i in range(len(probability)):
        unc = (-1 / np.log(n_classes)) * np.sum(probability[i] * np.log(_safe_prob(probability[i])))
        uncert.append(unc)
    return uncert


def entropy_modified(probability, target):
    entr_modi = []
    for i in range(len(probability)):
        ent_mod_1 = (-1) * (1 - probability[i][int(target[i])]) * np.log(_safe_prob(probability[i][int(target[i])]))
        probability_rest = np.delete(probability[i], int(target[i]))
        ent_mod_2 = -np.sum(probability_rest* np.log(_safe_prob(1 - probability_rest)))
        ent_mod = ent_mod_1 + ent_mod_2
        entr_modi.append(ent_mod)
    return entr_modi

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


import torch
import torch.nn as nn
import numpy as np
import copy

class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None, siadict=None,r=None,tracker=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users
        self.siadict = siadict
        self.r=r
        self.tracker = tracker

    def attack(self, net):
        confidence_all = []
        correct_loss = 0
        len_set = 0

        # Ensure model is on the correct device
        net.to(self.args.device)

        confidence_cov = []
        sia_per_client = []

        for idx in self.dict_mia_users:
            dataset_local = self.dict_mia_users[idx]                              
            y_loss_all = []

            # Evaluate each party's training data on each party's model
            for local in self.dict_mia_users:
                y_losse = []
                
                # Move index tensor to GPU
                idx_tensor = torch.tensor(idx, device=self.args.device)
                
                net.load_state_dict(self.w_locals[local])
                if self.args.method == 'fedalign':
                    net.apply(lambda m: setattr(m, 'width_mult', 1.0))
                net.eval()

                for id, (data, target) in enumerate(dataset_local):
                    data = data.to(self.args.device)  # Move data to GPU
                    target = target.to(self.args.device)  # Move target to GPU
                    
                    with torch.no_grad():  # No gradients needed for inference
                        log_prob = net(data)

                    # Compute prediction loss
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    
                    y_losse.append(y_loss.cpu().detach().numpy())  # Move loss to CPU before converting to NumPy

                y_losse = np.concatenate(y_losse).reshape(-1)
                y_loss_all.append(y_losse)

            # Convert y_loss_all to a tensor and move it to GPU
            y_loss_all = torch.tensor(y_loss_all, device=self.args.device)

            # Test if the owner party has the largest prediction probability
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]  # Keep on GPU

            # Compute correct_local_loss (moved everything to GPU before summation)
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))
            ).long().sum()

            # Move confidence values to CPU before converting to NumPy
            confidence = copy.deepcopy(y_loss_all.cpu().numpy())

            # Compute client confidence (average loss for each party)
            client_confidence = [np.mean(confidence_i) for confidence_i in confidence]
            confidence_cov.append(calculate_cv(client_confidence))
            confidence_all.append(client_confidence)

            self.tracker.add_sia_accuracy(self.r, idx, round(float(correct_local_loss.item() / len(dataset_local.dataset)), 2))

            sia_per_client.append(round(float(correct_local_loss / len(dataset_local.dataset)), 2))
            if idx in self.siadict:
                self.siadict[idx].append(round(float(correct_local_loss.item() / len(dataset_local.dataset)), 2))
            else:
                self.siadict[idx] = [round(float(correct_local_loss.item() / len(dataset_local.dataset)), 2)]

            correct_loss += correct_local_loss.item()
            len_set += len(dataset_local.dataset)

        # Calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set
        self.tracker.add_sia_accuracy_avg(self.r, accuracy_loss)
        self.tracker.add_confidence_all(self.r, confidence_all)
        print(f'\nLoss CoV: {np.mean(confidence_cov):.3f}, FI: {1 / (1 + np.square(np.mean(confidence_cov))):.3f}', )
        print(f'Sia CoV: {calculate_cv(sia_per_client):.3f}, FI: {1 / (1 + np.square(calculate_cv(sia_per_client))):.3f}')



        print('Average SIA attack accuracy : {}/{} ({:.2f}%)\n'.format(
            correct_loss, len_set, accuracy_loss
        ))
