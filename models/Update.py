import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .hessian import hessian # Hessian computation
import numpy as np

from torch.optim import Adam
import random


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

    def train(self, net, ):
        net.train()
        # train and update

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)  #lr=self.args.lr

        epoch_loss = []
        # SGD == 1
        for iter in range(self.args.local_ep):
        # for iter in range(1):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                if self.collab:
                    loss_lips = approximate_lipschitz(net)

                    # loss = loss/loss.detach() + self.lamb*(loss_lips/loss_lips.detach())
                    # loss = self.args.mu*(loss_CE/loss.item())*loss
                    # print('CE_loss & lips constant', loss.item(), loss_lips.item())
                    loss = loss + self.beta*self.lamb * (loss.item() / loss_lips.item())*loss_lips

                    # print('after norm', loss.item(), self.lamb * (loss.item() / loss_lips.item())*loss_lips.item())

                loss.backward()

                # gradient normalization?
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # create the hessian computation module
        # net.eval()
        if self.collab:
            hessian_comp = hessian(net, self.loss_func, dataloader=self.ldr_train, mps=True)
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=1)
            trace = hessian_comp.trace()
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
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), top_eigenvalues, np.mean(trace)
