'''
Main file to set up the FL system and train
Code design inspired by https://github.com/HongshengHu/SIAs-Beyond_MIAs_in_Federated_Learning
'''

import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics

import pickle

# we use prediction loss to conduct our attacks
# prediction loss: for a given sample (x, y), every local model will has a prediction loss on it. we consider the party who has the smallest prediction loss owns the sample.

# device = torch.device("mps")
device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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
        ent_mod_2 = -np.sum(probability_rest * np.log(_safe_prob(1 - probability_rest)))
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


##
global siadict, siaperclass
if 'siadict' not in globals():
    siadict = {}
if 'siaperclass' not in globals():
    siaperclass = []
##

def calculate_cv(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = (std_dev / mean) # * 100
    return cv


def inverse_softmax(x, axis=0):
    exp_x = torch.exp(-x)
    return exp_x / torch.sum(exp_x, dim=axis, keepdim=True)


class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_mia_users=None, flag=False):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_mia_users = dict_mia_users
        self.flag = flag
        # print('sia device', device)

    def attack(self, net, prediction_dic):
        correct_loss = 0
        len_set = 0

        confidence_all = []
        prediction_cnt = np.zeros(len(self.dict_mia_users))
        weighted_sel_cnt = np.zeros(len(self.dict_mia_users))
        sum_softmax_prob = []
        hit = []
        confidence_cov = []
        sia_per_client = []

        for idx in self.dict_mia_users:

            dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_mia_users[idx]),
                                       batch_size=self.args.local_bs, shuffle=False)

            # print(dataset_local)
            client_confidence = []

            y_loss_all = []
            client_data_cnt = np.zeros(len(self.dict_mia_users))

            # evaluate each party's training data on each party's model
            for local in self.dict_mia_users:

                y_losse = []

                idx_tensor = torch.tensor(idx)
                net.load_state_dict(self.w_locals[local])
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        # data, target = data.cuda(), target.cuda()
                        # idx_tensor = idx_tensor.cuda()
                        data = data.to(device)
                        target = target.to(device)
                        idx_tensor = idx_tensor.to(device)
                    log_prob = net(data)
                    # prediction loss based attack: get the prediction loss of the test sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    y_losse.append(y_loss.cpu().detach().numpy())

                y_losse = np.concatenate(y_losse).reshape(-1)

                y_loss_all.append(y_losse)

            # print('ylossall_lsit', y_loss_all)
            # y_loss_all = torch.tensor(y_loss_all).to(self.args.gpu)
            y_loss_all = torch.tensor(np.array(y_loss_all)).to(self.args.device) ###########

            # test if the owner party has the largest prediction probability
            # get the parties' index of the largest probability of each sample
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

            # confidence
            # print(y_loss_all.shape)
            # softmax loss
            # confidence = inverse_softmax(y_loss_all, axis=0).cpu().numpy()

            # RAW loss
            confidence = copy.deepcopy(y_loss_all.cpu().numpy())
            # print(confidence.shape)
            # print('len conf', len(confidence))
            # client confidence is the mean loss by using local model i
            client_confidence = [np.mean(confidence_i) for confidence_i in confidence]
            # confidence stats per client
            # print(f'client dataset {idx} confidence')
            # print(client_confidence)
            # print(calculate_cv(client_confidence))
            confidence_cov.append(calculate_cv(client_confidence))
            confidence_all.append(client_confidence)

            # softmax prediction
            prediction_prob = inverse_softmax(y_loss_all, axis=0).cpu().numpy()

            sum_softmax_prob.append(sum(prediction_prob[idx])/100)

            index_of_party_loss_cpu = copy.deepcopy(index_of_party_loss).cpu().numpy().flatten()
            for i in range(len(index_of_party_loss_cpu)):
                # print('index', index_of_party_loss_cpu)
                prediction_dic[index_of_party_loss_cpu[i]].append(prediction_prob[index_of_party_loss_cpu[i]][i])
            weighted_sel = 0
            hit_cnt = 0
            for i in range(len(index_of_party_loss_cpu)):
                if index_of_party_loss_cpu[i] == idx:
                    weighted_sel += prediction_prob[idx][i]
                    hit_cnt += 1
            # weighted_sel_cnt[idx] = weighted_sel/100
            if hit_cnt != 0:
                weighted_sel_cnt[idx] = weighted_sel / hit_cnt
                hit.append(hit_cnt)
            else:
                weighted_sel_cnt[idx] = 0
                hit.append(hit_cnt)


            for i in range(len(index_of_party_loss_cpu)):
                prediction_cnt[index_of_party_loss_cpu[i]] += 1
                client_data_cnt[index_of_party_loss_cpu[i]] += 1

            if self.flag:
                siaperclass.append(round(float(correct_local_loss / len(dataset_local.dataset)), 2))
            else:
                sia_per_client.append(round(float(correct_local_loss / len(dataset_local.dataset)), 2))
                if idx in siadict:
                    siadict[idx].append(round(float(correct_local_loss / len(dataset_local.dataset)), 2))
                else:
                    siadict[idx] = [round(float(correct_local_loss / len(dataset_local.dataset)), 2)]
            ##
            correct_loss += correct_local_loss
            len_set += len(dataset_local.dataset)

        # cov confidence for all clients
        # print('confidence_cov', confidence_cov)
        print(f'\nLoss CoV: {np.mean(confidence_cov):.3f}, FI: {1 / (1 + np.square(np.mean(confidence_cov))):.3f}', )
        print(f'Sia CoV: {calculate_cv(sia_per_client):.3f}, FI: {1 / (1 + np.square(calculate_cv(sia_per_client))):.3f}')

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        print('Average SIA attack accuracy : {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
                                                                                                  accuracy_loss))

        # print('sia_confidence_all', confidence_all)
        # if use losses:     np.mean(confidence_all, axis=0)
        # if use counters:   prediction_cnt/1000
        return accuracy_loss, confidence_all, prediction_cnt/1000, np.mean(confidence_all, axis=0), \
               weighted_sel_cnt, sum_softmax_prob, hit, prediction_dic # prediction_cnt/1000  np.mean(confidence_all, axis=0)#
