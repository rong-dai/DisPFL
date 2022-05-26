import copy
import logging
import math
import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
import torch.nn.functional as F






class cnn_cifar10_meta(nn.Module):

    # def random_growth(self, i, new_mask, weight):
    #     size = new_mask.size()
    #     new_mask=copy.deepcopy(new_mask).flatten()
    #     if self.num_remove[i] > 0:
    #         # weights with mask 0 has probability to regrow
    #         # logging.info("dense/numel {}/{}".format(torch.sum(mask[name]),torch.numel(mask[name] )))
    #         regrow_probability = 1 - new_mask.float()
    #         # randomly select  regrow index, and mask it to 1
    #         regrow_probability = regrow_probability.flatten()
    #         regrow_inx = torch.multinomial(regrow_probability, self.num_remove[i], replacement=False)
    #         new_mask[regrow_inx] = 1
    #     new_mask = new_mask.reshape(size)
    #     return new_mask
    #
    # # w: pruned weights
    # def magnitude_death(self, i, mask, weight ):
    #
    #     num_remove = math.ceil(self.drop_ratio * self.name2nonzeros[i])
    #     if num_remove == 0.0: return weight.data != 0.0
    #     num_zeros = self.name2zeros[i]
    #
    #     x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    #     n = idx.shape[0]
    #
    #     k = math.ceil(num_zeros + num_remove)
    #     threshold = x[k - 1].item()
    #
    #     return (torch.abs(weight.data) > threshold)

    def init_masks(self):
        dense_ratio = 0.2
        # dense_ratio = 1
        raw_conv_masks = {}
        for name, module in self.named_modules():
            if "meta" in name:
                raw_conv_masks[name+".weight"] = self.init_conv_masks( module.weight.shape, dense_ratio)
        return raw_conv_masks

    def init_conv_masks(self, size, dense_ratio):
        conv_mask = torch.zeros(size)
        dense_numel = int(dense_ratio * torch.numel(conv_mask))
        if dense_numel > 0:
            conv_mask = conv_mask.view(-1)
            perm = torch.randperm(len(conv_mask))
            perm = perm[:dense_numel]
            conv_mask[perm] = 1
            conv_mask = conv_mask.reshape(size)
        return conv_mask

    # def fit_pca(self, dense_ratio):
    #     train_num = 5000
    #     raw_conv_masks = []
    #     for i in range(train_num):
    #         mask_1 = self.init_conv_masks((64, 3, 5, 5),dense_ratio  ).flatten()
    #         mask_2 = self.init_conv_masks((64, 64, 5, 5),dense_ratio  ).flatten()
    #         concat_mask = torch.cat((mask_1, mask_2))
    #         raw_conv_masks.append(concat_mask)
    #     train_masks = torch.stack(raw_conv_masks, axis=0)
    #     pca = PCA(n_components=50)
    #     # logging.info("pca training started")
    #     transformer=pca.fit(train_masks)
    #     # logging.info("pca training finished")
    #     return transformer

    def __init__(self,  dense_ratio=0.2, used_meta=False):
        super(cnn_cifar10_meta, self).__init__()
        self.dense_ratio = dense_ratio
        self.used_meta =used_meta
        self.meta_conv1 = torch.nn.Conv2d(in_channels=3,
                                          out_channels=64,
                                          kernel_size=5,
                                          stride=1,
                                          padding=0, bias=False)
        self.meta_conv2 = torch.nn.Conv2d(64, 64, 5,bias=False)


        self.pool = torch.nn.MaxPool2d(kernel_size=3,
                                       stride=2)

        self.meta_fc1 = torch.nn.Linear(64 * 4 * 4, 10, bias=False)
        # self.fc2 = torch.nn.Linear(384, 192)
        # self.fc3 = torch.nn.Linear(192, 10)

    def get_masks(self):
        return self.raw_conv_masks

    def set_transformers(self, block_level_transformer):
        self.block_level_transformer = block_level_transformer



    # def _compress_conv_masks(self, device="cpu"):
    #     mask_1 = self.raw_conv_masks[0].flatten()
    #     mask_2 = self.raw_conv_masks[1].flatten()
    #     concat_mask = torch.cat((mask_1, mask_2))
    #     compressed_mask = self.block_level_transformer.transform(concat_mask.view(1, -1))
    #     return torch.tensor(compressed_mask.astype(np.float32)).squeeze().to(device)
    #     # update mask based on last forward

    # def next_mask(self, drop_ratio):
    #     self.update_drop_ratio(drop_ratio)
    #     conv_weights = self.meta_forward("cpu")
    #     self.name2nonzeros, self.name2zeros, self.num_remove = [[] for i in range(2)], [[] for i in range(2)],[[] for i in range(2)]
    #     next_mask = [[] for i in range(2)]
    #     self.num_remove = {}
    #     for i in range(2):
    #         mask = self.raw_conv_masks[i]
    #         self.name2nonzeros[i] = mask.sum().item()
    #         self.name2zeros[i] = mask.numel() - self.name2nonzeros[i]
    #         logging.info("before{}".format(torch.sum(mask)))
    #         next_mask[i] = self.magnitude_death(i, mask, conv_weights[i])
    #         logging.info("after{}".format(torch.sum(next_mask[i])))
    #         self.num_remove[i] = int(self.name2nonzeros[i] - next_mask[i].sum().item())
    #         next_mask[i] = self.random_growth(i, next_mask[i], conv_weights[i] )
    #     return next_mask

    def forward(self, x):

        x = self.pool(F.relu( self.meta_conv1(x)))
        x = self.pool(F.relu(self.meta_conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.meta_fc1(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

class Meta_net(nn.Module):
    def __init__(self, mask):
        super(Meta_net, self).__init__()
        size = int(mask.flatten().shape[0])
        # self.fc11 = nn.Linear(size, 200)
        # self.fc12 = nn.Linear(200, 200)
        # self.fc13 = nn.Linear(200, size)
        size = int(mask.flatten().shape[0])
        self.fc11 = nn.Linear(size, 50)
        self.fc12 = nn.Linear(50, 50)
        self.fc13 = nn.Linear(50, size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        x = F.relu(self.fc11(input.flatten()))
        x = F.relu(self.fc12(x))
        conv_weight = self.fc13(x).view(input.shape)
        return conv_weight

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)