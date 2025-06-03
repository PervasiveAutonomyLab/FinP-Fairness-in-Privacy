from .data_loader import *
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
def create_dict_mia_users(train_data_local_dict):
    new_dataloaders = {}

    for client_id, dataloader in train_data_local_dict.items():
        dataset = dataloader.dataset
        total_size = len(dataset)
        split_size = int(0.2 * total_size)
        remaining_size = total_size - split_size
        
        # Split the dataset
        subset1, subset2 = torch.utils.data.random_split(dataset, [remaining_size, split_size])
        
        # Create new DataLoader for the 20% subset
        new_dataloader = DataLoader(subset2, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers)
        
        # Store the new DataLoader
        new_dataloaders[client_id] = new_dataloader

    # Now new_dataloaders contains DataLoaders with 20% of the origi
    return new_dataloaders

    
def load_data(args):
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict,\
         class_num = load_partition_data(args.data_dir, args.partition_method, args.partition_alpha, args.client_number, args.batch_size)
    
    # Create a dictionary of DataLoaders with 20% of the data for each client
    mia_data_local_dict = create_dict_mia_users(train_data_local_dict)

    train_dataset = train_data_global.dataset
    test_dataset = test_data_global.dataset

     # Extract user distributions
    dict_party_user = {idx: list(loader.dataset.dataidxs) for idx, loader in train_data_local_dict.items()}
    # dict_test_use = {idx: list(loader.dataset.dataidxs) for idx, loader in test_data_local_dict.items()}
    dict_test_use = {
    idx: list(range(len(loader.dataset)))  # Generates [0, 1, ..., len(dataset)-1]
    for idx, loader in test_data_local_dict.items()
}


    #Sample from each user.
    dict_sample_user = {idx: list(loader.dataset.indices) for idx, loader in mia_data_local_dict.items()}
    return train_dataset, test_dataset, dict_party_user, dict_sample_user, dict_test_use, {}
    
