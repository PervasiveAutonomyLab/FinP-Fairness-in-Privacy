import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training") #20
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_samples', type=int, default=100,
                        help="number of samples from each local training set: N")
    parser.add_argument('--alpha', type=float, default=1, help="level of non-iid data distribution: alpha")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="testing batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate (default 0.01; 0.02 when --dataset FEMNIST)")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum for LocalUpdate")
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    # other arguments
    parser.add_argument('-c', '--checkpoint', default='checkpoint/synthetic', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--manualseed', type=int, default=42, help='manual seed')
    parser.add_argument('--dataset', type=str, default='Synthetic', help="name of dataset")
    parser.add_argument(
        '--femnist_hf_name',
        type=str,
        default='flwrlabs/femnist',
        help='HuggingFace dataset id for FEMNIST (when --dataset FEMNIST)',
    )
    parser.add_argument(
        '--femnist_test_fraction',
        type=float,
        default=0.2,
        help='Per-writer test fraction; train is 1 minus this (default 0.2 => 80%% train)',
    )
    parser.add_argument(
        '--femnist_remove',
        type=int,
        default=0,
        help='FEMNIST only: per client, randomly drop this many character labels (0..num_classes-1) '
        'and all their rows before train/test split; 0 = disabled (default)',
    )
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--all_clients', default=True, action='store_true', help='aggregation over all clients')
    parser.add_argument('--opt', default=False, action='store_true', help='using opt')
    parser.add_argument('--PCA', default=False, action='store_true', help='use PCA-based optimization for aggregation weights (requires --opt)')
    parser.add_argument('--col', default=False, action='store_true', help='collaboration')
    parser.add_argument('--resume', default=False, action='store_true', help='resume checkpoint')
    parser.add_argument('--beta', type=float, default=2, help="beta for lipchitz loss") # 0.3 for cifar best

    parser.add_argument('--runfed', default=False, action='store_true', help='using fedalign setup')


    parser.add_argument('--time_step', type=int, default=128, help='time_step')
    parser.add_argument('--time_channel', type=int, default=9, help='time_channel')


    ## fedalign
    parser.add_argument('--method', type=str, default='fedalign', metavar='N',
                help='Options are: fedavg, fedprox, moon, mixup, stochdepth, gradaug, fedalign')
    parser.add_argument('--data_dir', type=str, default='FedAlign/data/cifar10',
                        help='data directory: data/cifar100, data/cifar10, or another dataset')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)
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
    
    #dp
    parser.add_argument('--run_dp_baseline', action='store_true', help='Run the DP-SGD baseline instead of the custom method')
    parser.add_argument('--dp_noise', type=float, default=1.0, help='DP noise multiplier')
    parser.add_argument('--dp_clip', type=float, default=1.75, help='DP gradient clipping bound')
    parser.add_argument('--mia', action='store_true', help='Enable white-box per-round membership inference attack simulation')
    parser.add_argument('--hessian_eig_max_iter', type=int, default=100,
                        help='max power-iteration steps for top Hessian eigenvalue')
    parser.add_argument('--hessian_trace_max_iter', type=int, default=100,
                        help='max Hutchinson iterations for Hessian trace')
    parser.add_argument('--hessian_tol', type=float, default=1e-3,
                        help='tolerance used by Hessian eig/trace estimators')




    args = parser.parse_args()
    


    args.data_dir = 'data/cifar10'
    args.partition_alpha = 0.5
    args.client_number = 10

    if args.dataset == 'FEMNIST':
        args.num_classes = 62
        args.lr = 0.02

    return args
