import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from .sampling import sample_dirichlet_train_data
from .validationsets import testSubsets, per_class_subsets
from .har_data import process_har_data
# from .har_data_dirichlet import process_har_data
from .edu_data import process_edu_data
import pickle
import os
from .data_preprocessing import fedfl_adapter as fedad


def get_dataset(args):
    if args.dataset == 'HAR':
        train_dataset, test_dataset, dict_party_user, dict_sample_user, test_subsets, class_test, user_dict = \
            process_har_data(combine_train_test=True, balance_dataset=False, num_subjects=args.num_users,
                             random_activity_sample=True)    # , alpha_values=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        if args.opt and args.col:
            folder = 'results/HAR/finp/'
        elif args.opt:
            folder = 'results/HAR/opt/'
        elif args.col:
            folder = 'results/HAR/col/'
        else:
            folder = 'results/HAR/base/'
        filename = 'dataset_profile.pkl'
        file_path = os.path.join(folder, filename)
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # with open(file_path, 'wb+') as f:
        #     pickle.dump([user_dict, user_dict], f)
        test_subsets = list(test_subsets.values())
        class_test = list(class_test.values())
        return train_dataset, test_dataset, dict_party_user, dict_sample_user, test_subsets, class_test

    # if args.dataset == 'EDU':
    #     train_dataset, test_dataset, dict_party_user, dict_sample_user, test_subsets, class_test, user_dict = \
    #         process_edu_data(num_subjects=args.num_users, random_activity_sample=args.random_activity_sample)
    #     if args.RL is True:
    #         folder = 'results/EDU/'
    #     elif args.opt is True:
    #         folder = 'results/EDU/opt'
    #     else:
    #         folder = 'results/EDU/nonRL/'
    #     filename = 'dataset_profile.pkl'
    #     file_path = os.path.join(folder, filename)
    #     # file_path = './results/EDU/dataset_profile.pkl'
    #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    #     with open(file_path, 'wb+') as f:
    #         pickle.dump([user_dict, user_dict], f)
    #     test_subsets = list(test_subsets.values())
    #     class_test = list(class_test.values())
    #     return train_dataset, test_dataset, dict_party_user, dict_sample_user, test_subsets, class_test

    if args.dataset == 'CIFAR10':
        data_dir = './data/cifar/'
        ##### SIA data transform #####
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)
        
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
        #############################
        ###### FedAlign transform ######
        # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        #
        # train_transform = transforms.Compose([
        #     # transforms.ToPILImage(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        # ])
        #
        # valid_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        # ])
        # train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
        #                                  transform=train_transform)
        #
        # test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
        #                                 transform=valid_transform)
        ########################

        # sample non-iid data
        # dict_party_user, dict_sample_user = sample_dirichlet_train_data(train_dataset, args.num_users, args.num_samples,
        #                                                                args.alpha)
        ##################
        dict_party_user, dict_sample_user, actual_distribution, ideal_distribution = sample_dirichlet_train_data(
            train_dataset, args.num_users, args.num_samples,
            args.alpha)
        test_subsets = testSubsets(test_dataset, actual_distribution)
        class_test = per_class_subsets(test_dataset, actual_distribution)

        ##################
        ### res prepossing data###
        # return fedad.load_data(args)

        return train_dataset, test_dataset, dict_party_user, dict_sample_user, test_subsets, class_test



    else:
        train_dataset = []
        test_dataset = []
        test_subsets = []
        class_test = []
        dict_party_user, dict_sample_user = {}, {}
        print('+' * 10 + 'Error: unrecognized dataset' + '+' * 10)
    return train_dataset, test_dataset, dict_party_user, dict_sample_user, test_subsets, class_test


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : Adam')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'   Realistic dataset for Non-IID setting:{args.dataset}'f' has {args.num_classes} classes')
    print(f'   Level of non-iid data distribution:{args.alpha}')
    print(f'    Number of users  : {args.num_users}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
