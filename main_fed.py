'''
Main file to set up the FL system and train
Code design inspired by https://github.com/HongshengHu/SIAs-Beyond_MIAs_in_Federated_Learning
'''

import os
# os.environ['MKL_DISABLE_FAST_MM'] = '1'
import copy
import random
import shlex
import sys
import numpy as np
import torch
from models.Fed import *
from models.Sia import SIA, mean_pairwise_absdiff_over_n2, sen_welfare_from_reverse_scores, calculate_cv
from models.mia import (
    discover_attack_weight_keys,
    prepare_mia_static_data,
    run_whitebox_mia_round,
)
from models.Nets import CifarCnn, Conv1DCNN, FEMNISTNet
from models.Update import LocalUpdate, LocalUpdateDP
from models.test import test_img
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from utils.logger import Logger, mkdir_p
from utils.run_summary import print_federated_run_summary
from utils.PCAstate import flatten
import pickle
from models.Sia import siadict
from sklearn.decomposition import PCA
from scipy.optimize import minimize, Bounds
from collections import defaultdict
from models.resnet import resnet56 as resnet56_fedalign

import time

from FedAlign import *

import torchvision
from torch.utils.data import DataLoader
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

def find_optimal_clip_threshold(model, dataset, device, num_samples=200):
    """
    Estimates the optimal DP clipping threshold (C) by calculating the 
    90th percentile of per-sample gradient L2 norms.
    """
    print("\n[Diagnostic] Calculating empirical gradient norms...")
    
    # Put model in train mode to compute gradients
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    # CRITICAL: batch_size=1 ensures we get the exact gradient for a SINGLE record
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    grad_norms = []
    
    for i, (image, label) in enumerate(loader):
        if i >= num_samples:
            break
            
        image, label = image.to(device), label.to(device)
        model.zero_grad()
        
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        
        # Calculate the L2 norm of the gradients across all layers for this single image
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        grad_norms.append(total_norm)
        
    # Calculate key statistics
    p90_clip = np.percentile(grad_norms, 90)
    median_clip = np.median(grad_norms)
    max_clip = np.max(grad_norms)
    
    print(f"--- Gradient Norm Diagnostics ---")
    print(f"Analyzed {num_samples} individual records.")
    print(f"Median Norm:     {median_clip:.4f}")
    print(f"Maximum Norm:    {max_clip:.4f}")
    print(f"90th Percentile: {p90_clip:.4f}  <-- Recommended --dp_clip value")
    print("-" * 33 + "\n")
    
    return p90_clip


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


def print_fl_run_parameters(args, *, n_train=None, n_test=None):
    """Print resolved CLI / defaults and environment at startup (for logs / tee)."""
    lines = [
        "",
        "=" * 64,
        "  FinP main_fed — run parameters",
        "=" * 64,
    ]
    for k, v in sorted(vars(args).items()):
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        lines.append(f"  {k:<30} : {v}")
    if n_train is not None:
        lines.append(f"  {'dataset_train_len':<30} : {n_train}")
    if n_test is not None:
        lines.append(f"  {'dataset_test_len':<30} : {n_test}")
    lines.append(f"  {'local_optimizer (LocalUpdate)':<30} : SGD")
    if getattr(args, "opt", False) and getattr(args, "PCA", False):
        lines.append(f"  {'aggregation (--opt --PCA)':<30} : FedAvg with PCA-optimized weights")
    elif getattr(args, "opt", False):
        lines.append(f"  {'aggregation (--opt)':<30} : FedAvg with lambda-based weights")
    else:
        lines.append(f"  {'aggregation (not --opt)':<30} : FedAvg weighted by local train set size")
    lines.append(f"  {'collaboration (--col)':<30} : {bool(getattr(args, 'col', False))}")
    lines.append("=" * 64 + "\n")
    print("\n".join(lines))


def set_reproducible_seeds(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA). Does not enable strict deterministic CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_size_stats(model):
    """Compute parameter count and memory footprint (parameters + buffers)."""
    param_count = int(sum(p.numel() for p in model.parameters()))
    trainable_param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_bytes = int(
        sum(p.numel() * p.element_size() for p in model.parameters()) +
        sum(b.numel() * b.element_size() for b in model.buffers())
    )
    total_mib = float(total_bytes / (1024 ** 2))
    return {
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "total_bytes": total_bytes,
        "total_mib": total_mib,
    }


def build_model(args):
    """Construct the global model for the (model, dataset) pair.

    Returns ``(net_glob, deactivate)``. ``deactivate`` is True only for the
    ``--model res --dataset CIFAR10 --runfed`` path, where training is delegated
    to FedAlign (run here) and the standard federated loop is skipped.
    """
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        return CifarCnn(args=args).to(args.device), False
    if args.model == 'res' and args.dataset == 'CIFAR10':
        if args.runfed:
            fedalign_run(args)
            return None, True
        return resnet56_fedalign(class_num=10), False
    if args.model == 'tcn' and args.dataset == 'HAR':
        net_glob = Conv1DCNN(input_channels=args.time_channel, num_classes=args.num_classes)
        # Set the flatten dimension based on HAR input shape (9, 128)
        net_glob.set_flatten_dim(input_shape=(args.time_channel, args.time_step))
        return net_glob, False
    if args.model == 'femnistnet' and args.dataset == 'FEMNIST':
        return FEMNISTNet().to(args.device), False
    exit('Error: unrecognized model/dataset pair (e.g. use --model femnistnet --dataset FEMNIST)')


def setup_mia(args, net_glob, dataset_train, dict_party_user, rng_mia):
    """Prepare white-box MIA context when ``--mia`` is set; else return (None, [])."""
    if not getattr(args, "mia", False):
        return None, []
    if args.dataset != "FEMNIST" or args.model != "femnistnet":
        raise ValueError("--mia currently supports only --dataset FEMNIST with --model femnistnet.")
    mia_target_weight_keys = discover_attack_weight_keys(net_glob)
    mia_ctx = prepare_mia_static_data(args, dataset_train, dict_party_user, rng_mia)
    print(
        f"[MIA] Enabled. Dummy client=0; victims={len(mia_ctx['victim_ids'])}; "
        f"Writer A={mia_ctx['writer_a_id']}; Writer B={mia_ctx['writer_b_id']}; "
        f"layers={mia_target_weight_keys}"
    )
    return mia_ctx, mia_target_weight_keys


def resolve_results_path(args):
    """Return ``(folder, prefix)`` describing where pickled results are written.

    Layout: ``resultsdprun/<DATASET>/<method>/<sub>/`` where ``<sub>`` is
    ``noise_<n>_clip_<c>`` for DP baselines, otherwise ``beta_<beta>``. ``prefix``
    names the method (dp / finp / optonly / colonly / base) for the .pkl filename.
    """
    if getattr(args, 'run_dp_baseline', False):
        if args.dataset == 'CIFAR10':
            folder = 'resultsdprun/CIFAR/dp_res/' if args.model == 'res' else 'resultsdprun/CIFAR/dp/'
        elif args.dataset == 'HAR':
            folder = 'resultsdprun/HAR/dp/'
        elif args.dataset == 'FEMNIST':
            folder = 'resultsdprun/FEMNIST/dp/'
        else:
            exit('Error: unrecognized dataset for DP baseline results path')
        # Use DP hyper-parameters for the sub-folder instead of beta
        folder = f"{folder}noise_{args.dp_noise}_clip_{args.dp_clip}/"
        return folder, 'dp'

    # set the folder name
    if args.dataset == 'CIFAR10':
        if args.opt and args.col:
            folder = 'resultsdprun/CIFAR/finp_res/' if args.model == 'res' else 'resultsdprun/CIFAR/finp/'
        elif args.opt:
            folder = 'resultsdprun/CIFAR/opt_res/' if args.model == 'res' else 'resultsdprun/CIFAR/opt/'
        elif args.col:
            folder = 'resultsdprun/CIFAR/col_res/' if args.model == 'res' else 'resultsdprun/CIFAR/col/'
        else:
            folder = 'resultsdprun/CIFAR/base_res/' if args.model == 'res' else 'resultsdprun/CIFAR/base/'
    elif args.dataset == 'HAR':
        if args.opt and args.col:
            folder = 'resultsdprun/HAR/finp/'
        elif args.opt:
            folder = 'resultsdprun/HAR/opt/'
        elif args.col:
            folder = 'resultsdprun/HAR/col/'
        else:
            folder = 'resultsdprun/HAR/base/'
    elif args.dataset == 'FEMNIST':
        if args.opt and args.col:
            folder = 'resultsdprun/FEMNIST/finp/'
        elif args.opt:
            folder = 'resultsdprun/FEMNIST/opt/'
        elif args.col:
            folder = 'resultsdprun/FEMNIST/col/'
        else:
            folder = 'resultsdprun/FEMNIST/base/'
    else:
        exit('Error: unrecognized dataset for results path')

    folder = f"{folder}beta_{args.beta}/"

    if args.opt and args.col:
        prefix = 'finp'
    elif args.opt:
        prefix = 'optonly'
    elif args.col:
        prefix = 'colonly'
    else:
        prefix = 'base'
    return folder, prefix


def main():

    prediction_dic_list = []

    print(f"[Command] {' '.join(shlex.quote(arg) for arg in sys.argv)}")
    args = args_parser()
    set_reproducible_seeds(args.manualseed)
    print(f"Reproducibility: random / NumPy / PyTorch seeded with manualseed={args.manualseed}")
    # Keep independent RNG streams so enabling MIA does not perturb training sampling.
    rng_train = np.random.default_rng(seed=args.manualseed)
    rng_mia = np.random.default_rng(seed=args.manualseed + 1)

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
    print_fl_run_parameters(args, n_train=len(dataset_train), n_test=len(dataset_test))

    # build model
    net_glob, DEACTIVATE = build_model(args)

    if DEACTIVATE:
        # FedAlign already ran its own training inside build_model.
        return

    nn_size_stats = get_model_size_stats(net_glob)
    print(
        "[NN] Parameters: {:,} (trainable: {:,}) | Model size: {:.3f} MiB ({:,} bytes)".format(
            nn_size_stats["param_count"],
            nn_size_stats["trainable_param_count"],
            nn_size_stats["total_mib"],
            nn_size_stats["total_bytes"],
        )
    )
    # mia added
    mia_ctx, mia_target_weight_keys = setup_mia(args, net_glob, dataset_train, dict_party_user, rng_mia)

    #

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

        round_eod = []
        round_loss_cov = []
        round_loss_fi = []
        round_sia_cov = []
        round_sia_fi = []
        round_average_loss_mad = []
        round_sia_mad = []
        round_sen_welfare = []

        per_client_accuracy = []
        per_class_accuracy = []

        total_time = 0
        round_times = []
        pca_times = []
        hessian_avg_time_rounds = []
        mia_metrics_rounds = []
        mia_attack_accuracy_rounds = []
        mia_precision_rounds = []
        mia_recall_rounds = []
        mia_roc_auc_rounds = []
        mia_client_mad_rounds = []
        mia_reverse_sen_welfare_rounds = []
        mia_cov_rounds = []
        mia_fi_rounds = []
        mia_eod_rounds = []


        # Resolve where pickled results / checkpoints are written for this run.
        folder, prefix = resolve_results_path(args)
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

            if getattr(args, 'run_dp_baseline', False) and iter == 0:
                recommended_clip = find_optimal_clip_threshold(
                    model=copy.deepcopy(net_glob).to(args.device), 
                    dataset=dataset_train, 
                    device=args.device, 
                    num_samples=200
                )
                print(f"Recommended DP clipping threshold (C): {recommended_clip:.4f}")

            print('\n')
            print('Begin Round', iter)
            round_start_time = time.time()
            # mia added
            global_state_before_round = {
                k: v.detach().clone()
                for k, v in net_glob.state_dict().items()
            }
            #
            ##################
            # local training
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            # numbers of active client
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = rng_train.choice(range(args.num_users), m, replace=False)

            eigens = []
            traces = []
            hessian_times = []
            # print('lambs', lambs)

            checkpoint = {
                'epoch': iter,
                'global_model': net_glob.state_dict(),
                'client_models': {},
                'lambs': lambs
            }

            for idx in idxs_users:
                # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx], lamb=lambs[idx],
                #                     collab=args.col, beta=args.beta)

                # Check if we are running the DP Baseline comparison
                if getattr(args, 'run_dp_baseline', False):
                    local = LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_party_user[idx])
                else:
                    # Run YOUR custom method
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx], lamb=lambs[idx],
                                        collab=args.col, beta=args.beta)

                w, loss, eigenvalues, trace, hessian_time_sec = local.train(
                    copy.deepcopy(net_glob).to(args.device),
                    client_id=idx,
                    comm_round=iter,
                )

                checkpoint['client_models'][idx] = w

                eigens.append(float(eigenvalues[0]))
                traces.append(float(trace))
                hessian_times.append(float(hessian_time_sec))

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            hessian_eigen_history.append(eigens)
            hessian_trace_history.append(traces)
            avg_hessian_time = float(np.mean(hessian_times)) if len(hessian_times) > 0 else 0.0
            hessian_avg_time_rounds.append(avg_hessian_time)
            if args.col:
                value_r = pairwise(eigens)
                trace_r = pairwise(traces)

                lambs = (value_r + trace_r) / 2

            checkpoint['lambs'] = lambs
            os.makedirs(folder, exist_ok=True)
            torch.save(checkpoint, os.path.join(folder, f'resume_checkpoint.pth'))
            print(f'Checkpoint saved at epoch {iter}')
            print(
                "[Round {}] Hessian-only avg time/client: {:.4f}s | NN size: {:.3f} MiB".format(
                    iter,
                    avg_hessian_time,
                    nn_size_stats["total_mib"],
                )
            )
            # mia added
            if args.all_clients:
                client_to_weights = {cid: w_locals[cid] for cid in range(args.num_users)}
            else:
                client_to_weights = {int(cid): w_locals[pos] for pos, cid in enumerate(idxs_users)}

            if getattr(args, "mia", False):
                # Snapshot/restore torch RNG so MIA internals do not change training RNG progression.
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state_all = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                mia_round = run_whitebox_mia_round(
                    args=args,
                    rng=rng_mia,
                    global_model=net_glob,
                    global_state=global_state_before_round,
                    dataset_train=dataset_train,
                    dict_party_user=dict_party_user,
                    client_to_weights=client_to_weights,
                    mia_ctx=mia_ctx,
                    target_weight_keys=mia_target_weight_keys,
                )
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state_all is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state_all)
                mia_metrics_rounds.append(mia_round)
                if len(mia_round) > 0:
                    victim_ids_sorted = sorted(mia_round.keys())
                    acc_per_client = {cid: float(mia_round[cid]["accuracy"]) for cid in victim_ids_sorted}
                    acc_values = np.array([acc_per_client[cid] for cid in victim_ids_sorted], dtype=float)
                    prec_mean = float(np.mean([float(mia_round[cid]["precision"]) for cid in victim_ids_sorted]))
                    rec_mean = float(np.mean([float(mia_round[cid]["recall"]) for cid in victim_ids_sorted]))
                    auc_mean = float(np.mean([float(mia_round[cid]["roc_auc"]) for cid in victim_ids_sorted]))
                    acc_mean = float(np.mean(acc_values))

                    mia_attack_accuracy_rounds.append(acc_mean)
                    mia_precision_rounds.append(prec_mean)
                    mia_recall_rounds.append(rec_mean)
                    mia_roc_auc_rounds.append(auc_mean)
                    mia_mad = float(mean_pairwise_absdiff_over_n2(acc_values))
                    mia_client_mad_rounds.append(mia_mad)
                    reverse_mia = 1.0 - acc_values
                    reverse_sen_welfare = float(sen_welfare_from_reverse_scores(reverse_mia))
                    mia_reverse_sen_welfare_rounds.append(reverse_sen_welfare)

                    mia_cv_val = float(calculate_cv(acc_values))
                    if not np.isfinite(mia_cv_val):
                        mia_cv_val = float('nan')
                    mia_fi_val = float(1.0 / (1.0 + np.square(mia_cv_val))) if np.isfinite(mia_cv_val) else float('nan')
                    mia_eod_val = float(np.max(acc_values) - np.min(acc_values))
                    mia_cov_rounds.append(mia_cv_val)
                    mia_fi_rounds.append(mia_fi_val)
                    mia_eod_rounds.append(mia_eod_val)

                    per_client_str = ", ".join(
                        [f"client {cid}: {acc_per_client[cid]:.4f}" for cid in victim_ids_sorted]
                    )
                    print(f"[MIA][Round {iter}] Per-client accuracy -> {per_client_str}")
                    print(
                        "[MIA][Round {}] Victim mean metrics -> Acc: {:.4f}, Prec: {:.4f}, Recall: {:.4f}, ROC-AUC: {:.4f}".format(
                            iter,
                            acc_mean,
                            prec_mean,
                            rec_mean,
                            auc_mean,
                        )
                    )
                    print(
                        f"[MIA][Round {iter}] MIA client MAD: {mia_mad:.5f} | "
                        f"MIA CoV: {mia_cv_val:.4f} | MIA FI: {mia_fi_val:.4f} | MIA EOD: {mia_eod_val:.4f}"
                    )
                    print(
                        f"[MIA][Round {iter}] "
                        f"reverse_mia: {np.array2string(reverse_mia, precision=4, separator=', ')} | "
                        f"reverse_mia sen_welfare: {reverse_sen_welfare:.5f}"
                    )
                else:
                    mia_attack_accuracy_rounds.append(float("nan"))
                    mia_precision_rounds.append(float("nan"))
                    mia_recall_rounds.append(float("nan"))
                    mia_roc_auc_rounds.append(float("nan"))
                    mia_client_mad_rounds.append(float("nan"))
                    mia_reverse_sen_welfare_rounds.append(float("nan"))
                    mia_cov_rounds.append(float("nan"))
                    mia_fi_rounds.append(float("nan"))
                    mia_eod_rounds.append(float("nan"))
                    print(f"[MIA][Round {iter}] No victim metrics available.")

            #
            #####################################
            # implement source inference attack #
            #####################################
            prediction_dic = defaultdict(list)
            S_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_mia_users=dict_sample_user,
                        flag=False)
            attack_acc_loss, confidence_all, prediction_cnt, losses, weighted_sel_cnt, sum_softmax_prob, hit, prediction_dic, round_metrics \
                = S_attack.attack(net=empty_net.to(args.device), prediction_dic=prediction_dic)

            # print('sia acc loss', attack_acc_loss, confidence_all)
            sia_accuracy_rounds.append(attack_acc_loss.cpu().item())

            spc = round_metrics['sia_per_client']
            round_eod.append(float(max(spc) - min(spc)) if len(spc) > 0 else 0.0)
            round_loss_cov.append(round_metrics['loss_cov'])
            round_loss_fi.append(round_metrics['loss_fi'])
            round_sia_cov.append(round_metrics['sia_cov'])
            round_sia_fi.append(round_metrics['sia_fi'])
            round_average_loss_mad.append(round_metrics['average_loss_mad'])
            round_sia_mad.append(round_metrics['sia_mad'])
            round_sen_welfare.append(round_metrics['sen_welfare'])

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
            pca_weights = None
            if args.opt and args.PCA:
                pca_start_time = time.time()
                # Set initial guess as values of 0.1
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

                pca_weights = result.x
                w_opt.append(result.x)
                # print("Objective function value:", result.fun)
                pca_end_time = time.time()
                pca_times.append(pca_end_time - pca_start_time)
                # print(f"PCA time:{(pca_end_time - pca_start_time)}s = {(pca_end_time - pca_start_time) / 60}min")
            #############################################################


            # FedAvg: update global weights
            if args.opt and args.PCA:
                # PCA-based aggregation weights from scipy optimization
                w_glob = FedAvg(w_locals, pca_weights)
            elif args.opt is True:
                reversed_lamb_weights = 1 - np.array(lambs)
                lamb_weights = reversed_lamb_weights / sum(reversed_lamb_weights)
                w_glob = FedAvg(w_locals, lamb_weights)
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
                pickle.dump([round_times, pca_times, hessian_avg_time_rounds], f)



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
            pickle.dump([round_times, pca_times, hessian_avg_time_rounds, total_time], f)
        print('results saved in:',file_path)

        print_federated_run_summary(
            trainacc,
            testacc,
            sia_accuracy_rounds,
            round_eod,
            round_loss_cov,
            round_loss_fi,
            round_sia_cov,
            round_sia_fi,
            round_average_loss_mad,
            round_sia_mad,
            round_sen_welfare,
            mia_attack_accuracy_rounds=mia_attack_accuracy_rounds if getattr(args, "mia", False) else None,
            mia_precision_rounds=mia_precision_rounds if getattr(args, "mia", False) else None,
            mia_recall_rounds=mia_recall_rounds if getattr(args, "mia", False) else None,
            mia_roc_auc_rounds=mia_roc_auc_rounds if getattr(args, "mia", False) else None,
            mia_client_mad_rounds=mia_client_mad_rounds if getattr(args, "mia", False) else None,
            mia_reverse_sen_welfare_rounds=mia_reverse_sen_welfare_rounds if getattr(args, "mia", False) else None,
            mia_cov_rounds=mia_cov_rounds if getattr(args, "mia", False) else None,
            mia_fi_rounds=mia_fi_rounds if getattr(args, "mia", False) else None,
            mia_eod_rounds=mia_eod_rounds if getattr(args, "mia", False) else None,
            hessian_avg_time_rounds=hessian_avg_time_rounds,
            nn_size_mib=nn_size_stats["total_mib"],
            last_k=3,
        )


    ##
    # plotting
    # plotting(folder)


if __name__ == '__main__':
    main()

