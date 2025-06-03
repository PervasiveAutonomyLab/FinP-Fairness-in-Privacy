import torch
import numpy as np
import random
from collections import defaultdict
from .validationsets import testSubsets, per_class_subsets
from torch.utils.data import DataLoader, Subset

import pickle
import os

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
    rng = np.random.default_rng(seed=42)
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


    filename = 'dataset_profile.pkl'
    file_path = os.path.join('./test/', filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # with open('userdict.pkl', 'wb+') as f:
    #     pickle.dump(userdict, f)
    # with open('userdictdirichlet.pkl', 'wb+') as f:
    #     pickle.dump(userdictdirichlet, f)
    with open(file_path, 'wb+') as f:
        pickle.dump([userdict, userdictdirichlet], f)

    return per_participant_list, per_samples_list, userdict, userdictdirichlet


def sample_per_class(per_participant_list, data_classes):
    rng = np.random.default_rng(seed=42)
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

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from .old_sampling import sample_dirichlet_train_data
import pickle

def get_dataset(args):


    data_dir = './data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
    
    dict_party_user, dict_sample_user, actual_distribution, ideal_distribution = sample_dirichlet_train_data(
train_dataset, args.client_number, args.num_samples,
        args.alpha)
    
    test_subsets = testSubsets(test_dataset, actual_distribution)
    class_test = per_class_subsets(test_dataset, actual_distribution)

    num_classes = len(set([label for _, label in train_dataset]))
    num_samples_train_loader = len(train_dataset)
    num_samples_test_loader = len(test_dataset)



    client_train_DataLoader = {
        idx: DataLoader(Subset(train_dataset, indices), batch_size=args.batch_size, shuffle=True)
        for idx, indices in dict_party_user.items()
    }

    client_test_DataLoader = {
        idx: DataLoader(subset, batch_size=args.batch_size, shuffle=False)
        for idx, subset in enumerate(test_subsets)
    }

    client_Sample_DataLoader = {
        idx: DataLoader(Subset(train_dataset, indices), batch_size=args.batch_size, shuffle=True)
        for idx, indices in dict_sample_user.items()
    }
    client_train_count = {idx: len(indices) for idx, indices in dict_party_user.items()}


     # Convert train and test datasets to DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    class_test_loaders = [
        DataLoader(subset, batch_size=args.batch_size, shuffle=False) for subset in class_test
    ]

    print('+' * 10 + 'Error: unrecognized dataset' + '+' * 10)

    return num_samples_train_loader, num_samples_test_loader, train_loader, test_loader,client_train_count, client_train_DataLoader, client_test_DataLoader,num_classes, client_Sample_DataLoader, class_test_loaders 


