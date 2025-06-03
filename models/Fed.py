import copy
import numpy as np


def FedAvg(w, weight):
    w_avg = copy.deepcopy(w[0])
    # weight[0] = 0
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg

def FedAvgOpt(w, weight):
    w_avg = copy.deepcopy(w[0])
    # weight[0] = 0
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg


def FedAvgsel(w, weight, action):
    weight = np.copy(weight)
    # print('previous weight', weight)
    for i in range(len(action)):
        if action[i] == 0:
            weight[i] = 0
    # print('0 weight',weight)
    # weight[0] = 0
    weight = np.array(np.array(weight) / sum(weight))
    # print('norm weight', weight)

    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg


def FedAvgsim(w, weight):
    w_avg = copy.deepcopy(w[0])
    weight = np.full((1, len(w)), 1/len(w)).flatten()

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg


def FedAvg_d1(w, size_per_client, sia):
    # if len(sia[0]) == 1:
    idx = 0
    maxi = sia[0][0]
    for i in range(len(sia)):
        if sia[i][0] > maxi:
            maxi = sia[i][0]
            idx = i

    size_per_client[idx] = 0
    print('idx percentage', idx, sia[idx][0])
    print('sia', sia)
    total_size = sum(size_per_client)
    weight = np.array(np.array(size_per_client) / total_size)

    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg

def FedAvg_step(w, size_per_client, sia, deactls):
    ls = []
    for i in range(len(sia)):
        ls.append(sia[i][-1])
        # ls.append(sia[i][0])
    print(ls)
    srtls = copy.deepcopy(ls)
    srtls.sort(reverse=True)
    nlarge = srtls[:1]
    indx = []
    for i in range(len(nlarge)):
        if ls.index(nlarge[i]) not in deactls:
            deactls.append(ls.index(nlarge[i]))
    for j in range(len(deactls)):
        size_per_client[deactls[j]] = 0
    print('deactls', deactls)
        # print('sia', sia)
    print(indx)
    total_size = sum(size_per_client)
    weight = np.array(np.array(size_per_client) / total_size)

    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    # print('local weights', w)
    # print('glob w', w_avg)

    return w_avg, deactls

def FedAvg_dn(w, size_per_client, sia):
    # if len(sia[0]) == 1:
    ls = []
    for i in range(len(sia)):
        ls.append(sia[i][-1])
        # ls.append(sia[i][0])
    print(ls)
    srtls = copy.deepcopy(ls)
    srtls.sort(reverse=True)
    nlarge = srtls[:3]
    indx = []
    for i in range(len(nlarge)):
        indx.append(ls.index(nlarge[i]))
        size_per_client[ls.index(nlarge[i])] = 0



        # print('sia', sia)
    print(indx)
    total_size = sum(size_per_client)
    weight = np.array(np.array(size_per_client) / total_size)

    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * weight[0]
        for i in range(1, len(w)):
            w_avg[k] += weight[i] * w[i][k]

    return w_avg
