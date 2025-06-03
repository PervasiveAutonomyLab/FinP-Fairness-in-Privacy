#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import math


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.3):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer without activation
        return F.log_softmax(x, dim=1)  # Use log_softmax for classification


class Conv1DCNN(nn.Module):
    def __init__(self, input_channels=64, num_classes=6, dropout_prob=0.5):
        super(Conv1DCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        # Dropout layer
        ###
        # self.dropout = nn.Dropout(dropout_prob)
        # Fully connected layer (to be initialized with the correct flatten dimension later)
        self.flatten_dim = None  # Placeholder
        self.num_classes = num_classes
        self.fc = None

    def set_flatten_dim(self, input_shape):
        # Pass a dummy tensor to compute the flatten dimension
        x = torch.randn(1, *input_shape)  # Example input (batch, channels, sequence length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        self.flatten_dim = x.numel()  # Total elements in the flattened tensor
        self.fc = nn.Linear(self.flatten_dim, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        ###
        # x = self.dropout(x)  # Apply dropout
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Softmax activation for classification


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x

class Mnistcnn(nn.Module):
    def __init__(self, args):
        super(Mnistcnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class CHmnistcnn(nn.Module):
    def __init__(self, args):
        super(CHmnistcnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*13*13, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x




class CifarCnn(nn.Module):
    def __init__(self, args):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # out = F.max_pool2d(out, 2)
        out, _ = F.max_pool2d(out, 2, return_indices=True)
        out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        out, _ = F.max_pool2d(out, 2, return_indices=True)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
