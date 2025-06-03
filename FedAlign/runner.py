'''
Main file to set up the FL system and train
Code design inspired by https://github.com/mmendiet/FedAlign
'''

import pickle
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# Import data preprocessing modules
from .data_preprocessing import data_loader as dl
from .data_preprocessing import old_sampling as old_sampling
from .data_preprocessing import tracker_logs

import argparse

# Import model architectures
from .models import resnet56, resnet18
from .models import resnet56_gradaug
from .models import resnet18_gradaug
from .models import resnet56_stochdepth
from .models import resnet18_stochdepth
from .models import resnet56_fedalign
from .models import resnet18_fedalign

# Import Semantic Inversion Attack model
from .models import SIA

from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time
# from args import add_args
from collections import defaultdict

# Import federated learning methods
from .methods import fedavg as fedavg
from .methods import gradaug as gradaug
from .methods import fedprox as fedprox
from .methods import moon as moon
from .methods import stochdepth as stochdepth
from .methods import mixup as mixup
from .methods import fedalign as fedalign

def add_args(parser):
    """
    Add command line arguments for configuring the federated learning experiment.
    
    Args:
        parser: argparse.ArgumentParser object
        
    Returns:
        parsed arguments
    """
    # Training settings


    parser.add_argument('--data_dir', type=str, default='FedAlign/data/cifar10',
                        help='data directory: data/cifar100, data/cifar10, or another dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--alpha', type=float, default=0.5, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--num_users', type=int, default=10, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--local_ep', type=int, default=5, metavar='EP',
                        help='how many local_ep will be trained locally per round')

    parser.add_argument('--comm_round', type=int, default=20,
                        help='how many rounds of communications are conducted')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--mu', type=float, default=0.45, metavar='MU',
                        help='mu value for various methods')

    parser.add_argument('--width', type=float, default=0.25, metavar='WI',
                        help='minimum width for subnet training')

    parser.add_argument('--mult', type=float, default=1.0, metavar='MT',
                        help='multiplier for subnet training')

    parser.add_argument('--num_subnets', type=int, default=3,
                        help='how many subnets sampled during training')

    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=16, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--stoch_depth', default=0.5, type=float,
                    help='stochastic depth probability')

    parser.add_argument('--gamma', default=0.0, type=float,
                    help='hyperparameter gamma for mixup')
    args = parser.parse_args()

    return args

# Setup Functions
def set_random_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Helper Functions
def init_process(q, Client):
    """
    Initialize process for parallel client execution.
    
    Args:
        q: Queue containing client initialization information
        Client: Client class to instantiate
    """
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])

def run_clients(received_info):
    """
    Execute client training with provided information.
    
    Args:
        received_info: Information from server to client
        
    Returns:
        Client training results or None if interrupted
    """
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info('exiting')
        return None
    
def run_sia_attack(args, Model, client_weights, siadict, dict_mia_users, train_data_global, class_num, r, tracker):
    """
    Run Semantic Inversion Attack to evaluate privacy leakage.
    
    Args:
        args: Command line arguments
        Model: Model class for instantiation
        client_weights: Weights from clients for attack
        siadict: Dictionary to store SIA results
        dict_mia_users: Dictionary of membership inference data
        train_data_global: Global training dataset
        class_num: Number of classes in the dataset
        r: Current round number
        tracker: Metrics tracker
        
    Returns:
        The model used for attack
    """
    logging.info('Running SIA Attack')
    model = Model(class_num).to(args.device)
    w_locals = [x['weights'] for x in client_weights]
    S_attack = SIA(args=args, w_locals=w_locals, dataset=train_data_global, dict_mia_users=dict_mia_users, siadict=siadict, r=r, tracker=tracker)
    attack_acc_loss = S_attack.attack(model)

    return model

def allocate_clients_to_threads(args):
    """
    Allocate clients to threads for parallel execution.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary mapping thread IDs to lists of client IDs
    """
    mapping_dict = defaultdict(list)
    for round in range(args.epochs):
        # Determine which clients to use this round (sampling)
        if args.client_sample < 1.0:
            num_clients = int(args.num_users * args.client_sample)
            client_list = random.sample(range(args.num_users), num_clients)
        else:
            num_clients = args.num_users
            client_list = list(range(num_clients))
            
        # Distribute clients among threads
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t+clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict

def create_dict_mia_users(train_data_local_dict):
    """
    Create dictionary for membership inference attack by splitting each client's dataset.
    
    Args:
        train_data_local_dict: Dictionary of client training data
        
    Returns:
        Dictionary with 20% subset of each client's data for MIA
    """
    new_dataloaders = {}

    for client_id, dataloader in train_data_local_dict.items():
        dataset = dataloader.dataset
        total_size = len(dataset)
        split_size = int(0.2 * total_size)  # Take 20% for MIA
        remaining_size = total_size - split_size
        
        # Split the dataset
        subset1, subset2 = torch.utils.data.random_split(dataset, [remaining_size, split_size])
        
        # Create new DataLoader for the 20% subset
        new_dataloader = DataLoader(subset2, batch_size=dataloader.batch_size, 
                                  shuffle=True, num_workers=dataloader.num_workers)
        
        # Store the new DataLoader
        new_dataloaders[client_id] = new_dataloader

    return new_dataloaders

def fedalign_run(args):
    """
    Main function to run the federated learning experiment.
    Sets up the environment, initializes clients and server,
    and executes the training rounds.
    """
    # Initialize multiprocessing method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Set random seed for reproducibility
    set_random_seed()
    
    # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # args = add_args(parser)

    # args.method = set_method
    
    # Override some arguments for testing
    args.bs = args.batch_size
    args.device = 'cuda:0'
    args.data = 'cifar10'
    args.thread_number = args.num_users
    
    # Load and partition the dataset
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, class_num = dl.load_partition_data(
        args.data_dir, args.partition_method, args.alpha, args.num_users, args.batch_size)

    # Create data for membership inference attack
    dict_mia_users = create_dict_mia_users(train_data_local_dict)
    
    # Initialize metrics tracker
    tracker = tracker_logs.Tracker(args, class_num)
    
    # Analyze class distribution for each client
    client_class_distribution = {}
    for client_id, dataloader in train_data_local_dict.items():
        class_count = defaultdict(int)  # Count occurrences of each class
        
        for batch_idx, (x, targets) in enumerate(dataloader):
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            for label in targets:
                class_count[label] += 1
        
        client_class_distribution[client_id] = dict(class_count)

    # Map clients to threads for parallel execution
    mapping_dict = allocate_clients_to_threads(args)
    
    # Select appropriate method and model architecture based on args.method
    method_dict = {
        'fedavg': (fedavg.Server, fedavg.Client, resnet56, resnet18),
        'gradaug': (gradaug.Server, gradaug.Client, resnet56_gradaug, resnet18_gradaug),
        'fedprox': (fedprox.Server, fedprox.Client, resnet56, resnet18),
        'moon': (moon.Server, moon.Client, resnet56, resnet18),
        'stochdepth': (stochdepth.Server, stochdepth.Client, resnet56_stochdepth, resnet18_stochdepth),
        'mixup': (mixup.Server, mixup.Client, resnet56, resnet18),
        'fedalign': (fedalign.Server, fedalign.Client, resnet56_fedalign, resnet18_fedalign)
    }

    if args.method not in method_dict:
        raise ValueError('Invalid --method chosen! Please choose from available methods.')

    # Extract appropriate server, client and model classes
    Server, Client, Model_cifar, Model_other = method_dict[args.method]
    Model = Model_cifar if 'cifar' in args.data_dir else Model_other

    # Configure server
    server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 
                 'model_type': Model, 'num_classes': class_num}
                 
    # Configure clients
    client_dict = [{'train_data': train_data_local_dict, 'test_data': test_data_local_dict, 
                  'device': i % torch.cuda.device_count(),
                  'client_map': mapping_dict[i], 'model_type': Model, 
                  'num_classes': class_num} for i in range(args.thread_number)]

    # Add method-specific configurations
    if args.method in ['gradaug', 'fedalign']:
        width_range = [args.width, 1.0]
        # Different resolution settings depending on dataset
        resolutions = [32, 28, 24, 20] if 'cifar' in args.data_dir else [224, 192, 160, 128]
        if args.method == 'fedalign':
            resolutions = [32] if 'cifar' in args.data_dir else [224]
        # Add these settings to each client
        for client in client_dict:
            client['width_range'] = width_range
            client['resolutions'] = resolutions
    
    # Initialize clients
    client_info = []
    for i in range(len(client_dict)):
        client_info.append(Client(client_dict[i], args))

    # Set up log directory for saving results
    server_dict['save_path'] = '{}/logs/{}__{}_e{}_c{}'.format(
        os.getcwd(),
        time.strftime("%Y%m%d_%H%M%S"), 
        args.method, 
        args.local_ep, 
        args.num_users
    )
    
    if not os.path.exists(server_dict['save_path']):
        os.makedirs(server_dict['save_path'])
        
    # Initialize server
    server = Server(server_dict, args)
    server_outputs = server.start()

    # Dictionary to store semantic inversion attack results
    siadict = {}
    
    # Main training loop
    for r in range(args.epochs):
        logging.info(f'************** Round: {r} ***************')
        round_start = time.time()

        # Each client processes data and sends results back to the server
        client_outputs = []
        for i, client in enumerate(client_info):
            client_output = client.run(server_outputs[i])
            tracker.add_client_accuracy(r, i, client_output[0]['acc'])
            client_outputs.extend([x for x in client_output])
            logging.info(f'Client {i} finished processing data')
            
        # Run privacy attack to assess vulnerability
        run_sia_attack(args, Model, client_outputs, siadict, dict_mia_users, 
                     train_data_global, class_num, r, tracker)
                     
        # Send client outputs to server and get updated model
        server_outputs = server.run(client_outputs)
        
        # Track metrics
        tracker.add_model_accuracy_test(r, server.acc)
        tracker.add_model_accuracy_train(r, server.test(train_data_global))
        
        # Log round timing
        round_end = time.time()
        logging.info(f'Round {r} Time: {round_end - round_start:.2f}s')
        tracker.add_time(r, round_end - round_start)
        
    logging.info('Training Finished')
    
    # Save final results
    tracker.save_results("results/CIFAR/resnet/", filename=f'base-resnet-{args.method}.pkl')