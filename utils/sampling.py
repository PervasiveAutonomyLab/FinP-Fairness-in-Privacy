import torch
import numpy as np
import random
from collections import defaultdict
from .options import args_parser

import pickle
import os

args = args_parser()

# random.seed(args.manualseed)

def build_classes_dict(dataset):
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            label = label.numpy()[0]
        else:
            label = label
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]
    # print(classes[0])

    return classes



def sample_dirichlet_train_data(dataset, no_participants, no_samples, alpha=0.1):
    # random.seed(args.manualseed)
    rng = np.random.default_rng(seed=args.manualseed)
    data_classes = build_classes_dict(dataset)

    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list)
    no_classes = len(data_classes.keys())  # for cifar: 10

    global userdict, userdictdirichlet

    userdict = defaultdict(list)
    userdictdirichlet = defaultdict(list)

    for n in range(no_classes):
        image_num = []
        rng.shuffle(data_classes[n])
        sampled_probabilities = class_size * rng.dirichlet(
            np.array(no_participants * [alpha]))

        ###iid
        cap = len(data_classes[n])//no_participants
        # print(cap)

        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))

            ### iid
            # print(min(min(len(data_classes[n]), no_imgs), cap))
            sampled_list = data_classes[n][:min(min(len(data_classes[n]), no_imgs), cap)]
            ### noniid
            # sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]

            userdictdirichlet[user].append(no_imgs)
            ### iid
            userdict[user].append(min(min(len(data_classes[n]), no_imgs), cap))
            ### noniid
            # userdict[user].append(min(len(data_classes[n]), no_imgs))

            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            ### iid
            data_classes[n] = data_classes[n][min(min(len(data_classes[n]), no_imgs), cap):]
            ### non iid
            #data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]
    print('userdict', userdict)
    print('userdictdirichlet', userdictdirichlet)

    for i in range(len(per_participant_list)):
        no_samples = min(no_samples, len(per_participant_list[i]))

    for i in range(len(per_participant_list)):
        sample_index = rng.choice(len(per_participant_list[i]), no_samples,
                                        replace=False)
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])

    # sample_classes = build_classes_dict(dataset)
    # sample = sample_per_class(per_participant_list, sample_classes)

    # if args.dataset == 'MNIST':
    #     folder = './results/MNIST/'
    # elif args.dataset == 'CIFAR10':
    #     folder = './results/CIFAR/'
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


    filename = 'dataset_profile.pkl'
    file_path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # with open('userdict.pkl', 'wb+') as f:
    #     pickle.dump(userdict, f)
    # with open('userdictdirichlet.pkl', 'wb+') as f:
    #     pickle.dump(userdictdirichlet, f)
    with open(file_path, 'wb+') as f:
        pickle.dump([userdict, userdictdirichlet], f)

    return per_participant_list, per_samples_list, userdict, userdictdirichlet


def sample_per_class(per_participant_list, data_classes):
    rng = np.random.default_rng(seed=args.manualseed)
    #print(len(per_participant_list))
    classbook = defaultdict(list)
    for i in range(len(per_participant_list)):
        for j, idx in enumerate(per_participant_list[i]):
            for key, values in data_classes.items():
                #print(idx)
                # print(data_classes)
                if idx in values:
                    classbook[key].append(idx)
    # print(classbook)
    min_length = float('inf')  # Start with a very large number
    sample = defaultdict(list)
    # Iterate through the dictionary
    for key, value in classbook.items():
        if len(value) < min_length:
            min_length = len(value)

    for n in range(len(classbook)):
        rng.shuffle(classbook[n])
        sample[n].extend(classbook[n][:min_length])
    # print(len(sample))

    return sample

