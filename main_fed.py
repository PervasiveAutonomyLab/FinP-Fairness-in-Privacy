'''
Main file to set up the FL system and train
Code design inspired by https://github.com/HongshengHu/SIAs-Beyond_MIAs_in_Federated_Learning
'''

import os
# os.environ['MKL_DISABLE_FAST_MM'] = '1'
import copy
import numpy as np
import torch
from models.Fed import *
from models.Sia import SIA
from models.Nets import CifarCnn, Conv1DCNN
from models.Update import LocalUpdate
from models.test import test_img
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from utils.logger import Logger, mkdir_p
from utils.PCAstate import flatten
import pickle
from models.Sia import siadict
from sklearn.decomposition import PCA
from scipy.optimize import minimize, Bounds
from collections import defaultdict
from models.resnet import resnet56 as resnet56_fedalign

import time

from FedAlign import *


def pairwise(numbers):
    ranks = []
    for i in range(len(numbers)):
        tmp = 0
        for j in range(len(numbers)):
            tmp += abs(numbers[i]-numbers[j])
        ranks.append(tmp/(len(numbers)-1))
    ranks = [e/max(ranks) for e in ranks]
    return np.array(ranks)

def calculate_ssd(data):
    """
    Calculate the Sum of Squared Deviations (SSD) for a given list of numbers.

    Args:
    data (list or numpy.array): The input data.

    Returns:
    float: The Sum of Squared Deviations.
    """
    # Convert input to numpy array if it's not already
    data = np.array(data)

    # Calculate the mean
    mean = np.mean(data)

    # Calculate the deviations (differences from the mean)
    deviations = data - mean

    # Square the deviations
    squared_deviations = deviations ** 2

    # Sum the squared deviations
    ssd = np.sum(squared_deviations)

    return ssd


if __name__ == '__main__':

    prediction_dic_list = []

    args = args_parser()
    rng = np.random.default_rng(seed=args.manualseed)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # record the experimental results
    logger = Logger(os.path.join(args.checkpoint, 'log_seed{}.txt'.format(args.manualseed)))
    logger.set_names(['alpha', 'comm. round', 'ASR'])

    # parse args
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # args.device = torch.device("mps")
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print('DEVICE', args.device)
    # get training dataset, test dataset, 
    dataset_train, dataset_test, dict_party_user, dict_sample_user, test_subsets, per_class_test = get_dataset(args)

    DEACTIVATE = False

    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CifarCnn(args=args).to(args.device)
    elif args.model == 'res' and args.dataset == 'CIFAR10':
        if args.runfed:
            DEACTIVATE = True
            fedalign_run(args)
        else:
            DEACTIVATE = False
            no_class = 10
            net_glob = resnet56_fedalign(class_num=no_class)
    elif args.model == 'tcn' and args.dataset == 'HAR':
        net_glob = Conv1DCNN(input_channels=args.time_channel, num_classes=args.num_classes)
        # Set the flatten dimension based on HAR input shape (9, 128)
        net_glob.set_flatten_dim(input_shape=(args.time_channel, args.time_step))
    else:
        exit('Error: unrecognized model')


    if DEACTIVATE is False:
        empty_net = net_glob
        print('Net config')
        print(net_glob)
        net_glob.train()

        # per client dataset size and proportional weight
        size_per_client = []
        for i in range(args.num_users):
            size = len(dict_party_user[i])
            size_per_client.append(size)

        total_size = sum(size_per_client)
        size_weight = np.array(np.array(size_per_client) / total_size)

        # copy weights
        w_glob = net_glob.state_dict()

        # training
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []

        if args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]
        acc_loss_attack = []

        best_att_acc = 0

        deactls = []

        trainacc = []
        testacc = []

        reward_hist = []

        sia_repu = []

        pca = PCA(n_components=args.num_users)
        #pca_glob = PCA(n_components=args.num_users+1)

        conf = []
        pred = []
        pca_distance = []
        used_train_data = [[] for _ in range(args.num_users)]
        acc_diff_rounds = []
        sum_softmax_prob_rounds = []
        hit_rounds = []
        weighted_sel_cnt_rounds = []

        pca_ssd = []
        acc_diff_ssd = []
        w_opt = []

        hessian_eigen_history = []
        hessian_trace_history = []
        lambs = [0.5]*args.num_users    

        sia_accuracy_rounds = []
        sia_loss_rounds = []

        per_client_accuracy = []
        per_class_accuracy = []

        total_time = 0
        round_times = []
        pca_times = []


        # set the folder name
        if args.dataset == 'CIFAR10':
            if args.opt and args.col:
                if args.model == 'res':
                    folder = 'results/CIFAR/finp_res/'
                else:
                    folder = 'results/CIFAR/finp/'
            elif args.opt:
                if args.model == 'res':
                    folder = 'results/CIFAR/opt_res/'
                else:
                    folder = 'results/CIFAR/opt/'
            elif args.col:
                if args.model == 'res':
                    folder = 'results/CIFAR/col_res/'
                else:
                    folder = 'results/CIFAR/col/'
            else:
                if args.model == 'res':
                    folder = 'results/CIFAR/base_res/'
                else:
                    folder = 'results/CIFAR/base/'
        elif args.dataset == 'HAR':
            if args.opt and args.col:
                folder = 'results/HAR/finp/'
            elif args.opt:
                folder = 'results/HAR/opt/'
            elif args.col:
                folder = 'results/HAR/col/'
            else:
                folder = 'results/HAR/base/'


        if args.opt and args.col:
            prefix = 'finp'
        elif args.opt:
            prefix = 'optonly'
        elif args.col:
            prefix = 'colonly'
        else:
            prefix = 'base'
        filename = f'{prefix}_saved_results.pkl'
        file_path = os.path.join(folder, filename)
        print(file_path)


        time_start = time.time()

        resume_round = 0
        # if resume
        if args.resume:
            # resume from checkpoint
            resume_point = torch.load(os.path.join(folder, 'resume_checkpoint.pth'), weights_only=False)
            print(f'resume from check point {resume_point['epoch']}')
            net_glob.load_state_dict(resume_point['global_model'])
            resume_round = resume_point['epoch']
            lambs = resume_point['lambs']

            filename = f'{prefix}_saved_results_resume_from_{resume_round}.pkl'
            file_path = os.path.join(folder, filename)
            print(file_path)

        # start training
        for iter in range(resume_round, args.epochs):
            print('\n')
            print('Begin Round', iter)
            round_start_time = time.time()
            ##################
            # local training
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            # numbers of active client
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = rng.choice(range(args.num_users), m, replace=False)

            eigens = []
            traces = []
            # print('lambs', lambs)

            checkpoint = {
                'epoch': iter,
                'global_model': net_glob.state_dict(),
                'client_models': {},
                'lambs': lambs
            }

            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx], lamb=lambs[idx],
                                    collab=args.col, beta=args.beta)

                w, loss, eigenvalues, trace = local.train(net=copy.deepcopy(net_glob).to(args.device))

                checkpoint['client_models'][idx] = w

                eigens.append(float(eigenvalues[0]))
                traces.append(float(trace))

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            hessian_eigen_history.append(eigens)
            hessian_trace_history.append(traces)
            if args.col:
                value_r = pairwise(eigens)
                trace_r = pairwise(traces)

                lambs = (value_r + trace_r) / 2

            checkpoint['lambs'] = lambs
            os.makedirs(folder, exist_ok=True)
            torch.save(checkpoint, os.path.join(folder, f'resume_checkpoint.pth'))
            print(f'Checkpoint saved at epoch {iter}')


            #####################################
            # implement source inference attack #
            #####################################
            prediction_dic = defaultdict(list)
            S_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_mia_users=dict_sample_user,
                        flag=False)
            attack_acc_loss, confidence_all, prediction_cnt, losses, weighted_sel_cnt, sum_softmax_prob, hit, prediction_dic \
                = S_attack.attack(net=empty_net.to(args.device), prediction_dic=prediction_dic)

            # print('sia acc loss', attack_acc_loss, confidence_all)
            sia_accuracy_rounds.append(attack_acc_loss.cpu().item())

            sia_loss_rounds.append(confidence_all)
            # print('all', sia_accuracy_rounds, sia_loss_rounds)

            prediction_dic_list.append(prediction_dic)
            sum_softmax_prob_rounds.append(sum_softmax_prob)
            hit_rounds.append(hit)
            weighted_sel_cnt_rounds.append(weighted_sel_cnt)
            conf.append(losses)
            pred.append(prediction_cnt)
            # print('cnt&loss ssd', calculate_ssd(prediction_cnt), calculate_ssd(losses))

            logger.append([args.alpha, iter, attack_acc_loss])

            # save model for the epoch that achieve the max source inference accuracy
            if attack_acc_loss > best_att_acc:
                torch.save(w_locals, os.path.join(args.checkpoint, 'model_weight'))
                torch.save(dict_party_user, os.path.join(args.checkpoint, 'local_data'))

            best_att_acc = max(best_att_acc, attack_acc_loss)

            ###################
            ### scipy.opt #####
            ###################
            if args.opt:
                pca_start_time = time.time()
                # Set initial guess as 10 values of 0.1
                x0 = np.full(args.num_users, 0.1)

                def objective_function(x, w):
                    w_glob_ = FedAvgOpt(w, x)
                    pca_s = []
                    pca_s.append(flatten(copy.deepcopy(w_glob_)))

                    for n in range(args.num_users):
                        pca_s.append(flatten(w[n]))
                    pca_trans_res = pca.fit_transform(pca_s)
                    pca_d = []
                    for p in range(1, len(pca_trans_res)):
                        pca_d.append(np.linalg.norm(np.array(pca_trans_res[p]) - np.array(pca_trans_res[0])))
                    finp = calculate_ssd(pca_d)
                    return finp


                # Define constraints
                def constraint_sum_to_one(x):
                    return np.sum(x) - 1.0
                constraint = {'type': 'eq', 'fun': constraint_sum_to_one}
                bounds = Bounds(np.zeros(args.num_users), np.ones(args.num_users))

                # Perform optimization
                result = minimize(objective_function, x0, args=(w_locals,), method='SLSQP',
                                bounds=bounds, constraints=constraint)   

                w_opt.append(result.x)
                # print("Objective function value:", result.fun)
                pca_end_time = time.time()
                pca_times.append(pca_end_time - pca_start_time)
                # print(f"PCA time:{(pca_end_time - pca_start_time)}s = {(pca_end_time - pca_start_time) / 60}min")
            #############################################################


            # FedAvg: update global weights
            if args.opt is True:
                w_glob = FedAvg(w_locals, result.x)
            else:
                w_glob = FedAvg(w_locals, size_weight)

            # testing
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # testing accuracy(using dataset_test, overall acc) ###########
            acc_train, loss_train_ = test_img(net_glob, dataset_train, args)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Average training loss {:.5f}'.format(loss_avg))
            loss_train.append(loss_avg)
            print(f'training accuracy: {float(acc_train):.2f}%')
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print(f'testing accuracy: {float(acc_test):.2f}%')
            testacc.append(float(acc_test))
            trainacc.append(float(acc_train))


            l_w = copy.deepcopy(w_locals)


            round_end_time = time.time()
            round_times.append(round_end_time - round_start_time)
            print(f"Round time: {(round_end_time - round_start_time):.3f}s = {(round_end_time - round_start_time) / 60:.3f}min")

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb+') as f:
                pickle.dump([siadict, trainacc, testacc, sia_loss_rounds, sia_accuracy_rounds, round_times], f)

            filename_time = f'{prefix}_saved_results_time.pkl'
            file_path_time = os.path.join(folder, filename_time)
            os.makedirs(os.path.dirname(file_path_time), exist_ok=True)
            with open(file_path_time, 'wb+') as f:
                pickle.dump([round_times, pca_times], f)



        logger.close()

        # testing after epochs finished
        net_glob.eval()
        exp_details(args)
        acc_train, loss_train_ = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        print("Training accuracy: {:.2f} %".format(acc_train))
        print("Testing accuracy: {:.2f} %".format(acc_test))
        print('Best attack accuracy: {:.2f} %'.format(best_att_acc))

        time_end = time.time()
        total_time = time_end - time_start
        print(f"Total time: {total_time}s = {(total_time) / 60}min = {total_time / 3600}h")

        filename_time_total = f'{prefix}_saved_results_time_total.pkl'
        file_path_time_total = os.path.join(folder, filename_time_total)
        os.makedirs(os.path.dirname(file_path_time_total), exist_ok=True)
        with open(file_path_time_total, 'wb+') as f:
            pickle.dump([round_times, pca_times, total_time], f)


    ##
    # plotting
    # plotting(folder)

