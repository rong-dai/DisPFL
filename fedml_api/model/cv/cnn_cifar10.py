import copy
import logging
import math
import random

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
track_running_stats=False
class cnn_cifar10(nn.Module):
    def __init__(self):
        super(cnn_cifar10, self).__init__()
        self.n_cls = 10
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = torch.nn.Linear(64*5*5, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class cnn_cifar100(nn.Module):
    def __init__(self):
        super(cnn_cifar100, self).__init__()
        self.n_cls = 100
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = torch.nn.Linear(64*5*5, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class cnn_cifar10(nn.Module):
#
#     def __init__(self):
#         super(cnn_cifar10, self).__init__()
#         self.conv1 = torch.nn.Conv2d(in_channels=3,
#                                           out_channels=64,
#                                           kernel_size=5,
#                                           stride=1,
#                                           padding=0, bias=False)
#         self.conv2 = torch.nn.Conv2d(64, 64, 5,bias=False)
#         self.pool = torch.nn.MaxPool2d(kernel_size=3,
#                                        stride=2)
#         self.fc1 = torch.nn.Linear(64 * 4 * 4, 10, bias=False)
#         # self.fc2 = torch.nn.Linear(384, 192)
#         # self.fc3 = torch.nn.Linear(192, 10)
#
#
#     def forward(self, x):
#         x = self.pool(F.relu( self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 4 * 4)
#         x = self.fc1(x)
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         return x

# class Meta_net(nn.Module):
#     def __init__(self, mask):
#         super(Meta_net, self).__init__()
#         size = int(mask.flatten().shape[0])
#         # self.fc11 = nn.Linear(size, 200)
#         # self.fc12 = nn.Linear(200, 200)
#         # self.fc13 = nn.Linear(200, size)
#         size = int(mask.flatten().shape[0])
#         self.fc11 = nn.Linear(size, 50)
#         self.fc12 = nn.Linear(50, 50)
#         self.fc13 = nn.Linear(50, size)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight.data)
#                 nn.init.constant_(m.bias.data, 0)
#
#     def forward(self, input):
#         x = F.relu(self.fc11(input.flatten()))
#         x = F.relu(self.fc12(x))
#         conv_weight = self.fc13(x).view(input.shape)
#         return conv_weight
#
#     def initialize_weights(m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight.data, 1)
#             nn.init.constant_(m.bias.data, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight.data)
#             nn.init.constant_(m.bias.data, 0)