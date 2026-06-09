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


def mean_pairwise_absdiff_over_n2(xs):
    """ΣΣ |x_i - x_j| / n² over all ordered pairs (i, j), n = len(xs)."""
    x = np.asarray(xs, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    return float(np.sum(np.abs(x[:, np.newaxis] - x[np.newaxis, :])) / (n * n))


def sen_welfare_from_reverse_scores(scores):
    """
    Sensitivity-aware welfare:
    mu * (1 - pair_sum / (2 * n^2 * mu)), where scores are reverse metrics.
    """
    arr = np.asarray(scores, dtype=float)
    n = arr.size
    if n == 0:
        return 0.0
    mu = float(np.mean(arr))
    if abs(mu) < 1e-15:
        return 0.0
    pair_sum = float(np.sum(np.abs(arr[:, np.newaxis] - arr[np.newaxis, :])))
    return float(mu * (1.0 - (pair_sum / (2.0 * (n ** 2) * mu))))


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
        confidence_mad = []
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
            confidence_mad.append(mean_pairwise_absdiff_over_n2(client_confidence))
            confidence_all.append(client_confidence)

            # softmax prediction
            prediction_prob = inverse_softmax(y_loss_all, axis=0).cpu().numpy()

            # Mean over this victim's MIA subset (aligned with --num_samples / actual |dict_sample_user[idx]|)
            n_mia = max(1, len(dataset_local.dataset))
            sum_softmax_prob.append(float(np.sum(prediction_prob[idx])) / n_mia)

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
        print(f'confidence_mad (per victim): {confidence_mad}')
        average_loss_mad = float(np.mean(confidence_mad))
        print(f'average_loss_mad: {average_loss_mad:.5f}')
        sia_mad = float(mean_pairwise_absdiff_over_n2(sia_per_client)) if sia_per_client else float('nan')
        print(f'sia_mad: {sia_mad:.5f}')
        print(f'Sia CoV: {calculate_cv(sia_per_client):.3f}, FI: {1 / (1 + np.square(calculate_cv(sia_per_client))):.3f}')

        reverse_sia = np.array([1.0 - float(s) for s in sia_per_client], dtype=float)
        sen_welfare = sen_welfare_from_reverse_scores(reverse_sia)
        print(f'reverse_sia: {reverse_sia}')
        print(f'sen_welfare: {sen_welfare:.5f}')

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        print('Average SIA attack accuracy : {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
                                                                                                  accuracy_loss))

        # Normalized argmin-win counts (sums to 1); replaces fixed /1000 = 1/num_users/num_samples assumption
        total_assign = float(np.sum(prediction_cnt))
        norm_prediction_cnt = prediction_cnt / max(total_assign, 1.0)

        loss_cov_mean = float(np.mean(confidence_cov)) if confidence_cov else float('nan')
        loss_fi_val = float(1.0 / (1.0 + np.square(loss_cov_mean))) if np.isfinite(loss_cov_mean) else float('nan')
        if sia_per_client:
            sia_cv_val = float(calculate_cv(sia_per_client))
            if not np.isfinite(sia_cv_val):
                sia_cv_val = float('nan')
            sia_fi_val = float(1.0 / (1.0 + np.square(sia_cv_val))) if np.isfinite(sia_cv_val) else float('nan')
        else:
            sia_cv_val = float('nan')
            sia_fi_val = float('nan')

        round_metrics = {
            'loss_cov': loss_cov_mean,
            'loss_fi': loss_fi_val,
            'sia_cov': sia_cv_val,
            'sia_fi': sia_fi_val,
            'average_loss_mad': float(average_loss_mad),
            'sia_mad': float(sia_mad),
            'sen_welfare': float(sen_welfare),
            'sia_per_client': list(sia_per_client),
        }

        return accuracy_loss, confidence_all, norm_prediction_cnt, np.mean(confidence_all, axis=0), \
               weighted_sel_cnt, sum_softmax_prob, hit, prediction_dic, round_metrics
