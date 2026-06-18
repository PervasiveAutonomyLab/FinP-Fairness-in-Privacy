import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time

from .hessian import hessian # Hessian computation
import numpy as np

import random
from opacus import PrivacyEngine


class CollectedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def approximate_lipschitz(model):
    lipschitz_constant = torch.tensor(1.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            weight = module.weight
            if weight is not None:
                # spectral_norm = torch.linalg.norm(weight.view(weight.size(0), -1), ord=2)
                spectral_norm = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1).max()
                lipschitz_constant *= spectral_norm
            else:
                return None
        elif isinstance(module, nn.ReLU):
            lipschitz_constant *= 1.0
        elif isinstance(module, nn.Sigmoid):
            lipschitz_constant *= 0.25
        elif isinstance(module, nn.Tanh):
            lipschitz_constant *= 1.0
        elif isinstance(module, nn.MaxPool1d):
            # MaxPool1d preserves the Lipschitz constant
            pass
        elif isinstance(module, nn.Dropout):
            # Dropout during training 1/(1-p) during inference 1-p
            # p=0.5
            # lipschitz_constant *= 2.0
            pass
    return lipschitz_constant


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, lamb=0.5, collab=False, beta=0.5):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lamb = lamb
        self.collab = collab
        self.beta = beta
        # self.used_data = []

    def train(self, net, client_id=None, comm_round=None):
        net.train()
        # train and update

        # SGD for all datasets (lr and momentum from args; FEMNIST+femnistnet lr set in main_fed.py).
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

        epoch_loss = []
        # SGD == 1
        for iter in range(self.args.local_ep):
        # for iter in range(1):
            batch_loss = []
            epoch_total_sum = 0.0
            epoch_ce_sum = 0.0
            epoch_extra_sum = 0.0
            epoch_correct = 0
            epoch_total = 0
            n_batches = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)

                ce = self.loss_func(log_probs, labels)
                loss = ce
                if self.collab:
                    loss_lips = approximate_lipschitz(net)

                    # loss = loss/loss.detach() + self.lamb*(loss_lips/loss_lips.detach())
                    # loss = self.args.mu*(loss_CE/loss.item())*loss
                    # print('CE_loss & lips constant', loss.item(), loss_lips.item())
                    loss = loss + self.beta*self.lamb * (loss.item() / loss_lips.item())*loss_lips

                    # print('after norm', loss.item(), self.lamb * (loss.item() / loss_lips.item())*loss_lips.item())

                with torch.no_grad():
                    pred = log_probs.argmax(dim=1)
                    epoch_correct += (pred == labels).sum().item()
                    epoch_total += labels.size(0)
                    extra = (loss - ce).detach().item()
                epoch_ce_sum += ce.item()
                epoch_extra_sum += extra
                epoch_total_sum += loss.item()
                n_batches += 1

                loss.backward()

                # gradient normalization?
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

            mean_total = epoch_total_sum / max(n_batches, 1)
            mean_ce = epoch_ce_sum / max(n_batches, 1)
            mean_extra = epoch_extra_sum / max(n_batches, 1)
            acc_pct = 100.0 * epoch_correct / max(epoch_total, 1)
            prefix = ""
            if comm_round is not None and client_id is not None:
                prefix = f"[Round {comm_round}] Client {client_id} | "
            print(
                f"  {prefix}Local Epoch {iter + 1}/{self.args.local_ep} | "
                f"Total loss: {mean_total:.4f} | CE: {mean_ce:.4f} | Lipschitz term: {mean_extra:.4f} | "
                f"Acc: {acc_pct:.2f}%"
            )

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # create the hessian computation module
        # net.eval()
        hessian_time_sec = 0.0
        if self.collab:
            if torch.cuda.is_available() and str(self.args.device).startswith("cuda"):
                torch.cuda.synchronize()
            hessian_t0 = time.perf_counter()
            hessian_comp = hessian(
                net, self.loss_func, dataloader=self.ldr_train, device=self.args.device
            )
            top_eigenvalues, _ = hessian_comp.eigenvalues(
                top_n=1,
                maxIter=getattr(self.args, "hessian_eig_max_iter", 100),
                tol=getattr(self.args, "hessian_tol", 1e-3),
            )
            trace = hessian_comp.trace(
                maxIter=getattr(self.args, "hessian_trace_max_iter", 100),
                tol=getattr(self.args, "hessian_tol", 1e-3),
            )
            if torch.cuda.is_available() and str(self.args.device).startswith("cuda"):
                torch.cuda.synchronize()
            hessian_time_sec = time.perf_counter() - hessian_t0
        else:
            top_eigenvalues = [random.randint(0, 9)]
            trace = random.randint(0, 9)

        # debugging
        # top_eigenvalues = [random.randint(0, 9)]
        # trace = random.randint(0, 9)

        # for batch_idx, (images, labels) in enumerate(self.ldr_train):
        #     self.used_data.append((images, labels))
        #     print(images, labels)
        # print('ldr_train', self.ldr_train[0])
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), top_eigenvalues, np.mean(trace), float(hessian_time_sec)



#### DP UPDATE
class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, client_id=None, comm_round=None):
        net.train()
        
        # Standard DP baselines typically use SGD with momentum rather than Adam
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)

        # Apply Differential Privacy
        privacy_engine = PrivacyEngine()
        net, optimizer, self.ldr_train = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=self.ldr_train,
            noise_multiplier=self.args.dp_noise, # e.g., 1.0
            max_grad_norm=self.args.dp_clip,     # e.g., 1.2
        )

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            epoch_total_sum = 0.0
            epoch_ce_sum = 0.0
            epoch_correct = 0
            epoch_total = 0
            n_batches = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                with torch.no_grad():
                    pred = log_probs.argmax(dim=1)
                    epoch_correct += (pred == labels).sum().item()
                    epoch_total += labels.size(0)
                ce_val = loss.item()
                epoch_ce_sum += ce_val
                epoch_total_sum += ce_val
                n_batches += 1

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            mean_total = epoch_total_sum / max(n_batches, 1)
            mean_ce = epoch_ce_sum / max(n_batches, 1)
            acc_pct = 100.0 * epoch_correct / max(epoch_total, 1)
            prefix = ""
            if comm_round is not None and client_id is not None:
                prefix = f"[Round {comm_round}] Client {client_id} | "
            print(
                f"  {prefix}Local Epoch {iter + 1}/{self.args.local_ep} | "
                f"Total loss: {mean_total:.4f} | CE: {mean_ce:.4f} | Lipschitz term: {0.0:.4f} | "
                f"Acc: {acc_pct:.2f}%"
            )

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Safely unwrap the model to extract the standard state_dict
        unwrapped_net = net._module

        # Return state_dict, loss, and dummy values for the Hessian/trace 
        # so it perfectly matches the return signature of your custom LocalUpdate
        return unwrapped_net.state_dict(), sum(epoch_loss) / len(epoch_loss), [0.0], 0.0, 0.0