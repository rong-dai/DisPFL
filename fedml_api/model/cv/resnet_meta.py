'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''
import logging

import torch
import torch.nn as nn
import torch.functional as F
__all__ = ['ResNet', 'resnet110']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""


    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.meta_conv1 = conv1x1(cfg[0], cfg[1])
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.meta_conv2 = conv3x3(cfg[1], cfg[1],  )
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.meta_conv3 = conv1x1(cfg[2], cfg[2]*expansion)

        if self.downsample is not None:
            self.meta_conv_ds = conv1x1(self.inplanes, self.oup_channels, stride),
            self.bn_ds = nn.BatchNorm2d(self.oup_channels, affine=False)


    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.meta_conv_ds(x)
            identity = self.bn_ds(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, block_num, num_classes=10, zero_init_residual=False, norm_layer=None, KD=False):
        super(ResNet, self).__init__()

        cfg = [[16, 16, 16], [64, 16, 16] * (n - 1), [64, 32, 32], [128, 32, 32] * (n - 1), [128, 64, 64],
               [256, 64, 64] * (n - 1), [256]]
        cfg = [item for sub_list in cfg for item in sub_list]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        layers=[]
        self._make_layer(block, 16, block_num[0],layers)
        self._make_layer(block, 32, block_num[1],layers, stride=2)
        self._make_layer(block, 64, block_num[2], layers, stride=2)
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    def _make_layer(self, block, planes, blocks, layers):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         norm_layer(planes * block.expansion),
        #     )
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        for layer in self.layers:
            x =  layer(x)
        x = self.avgpool(x)
        x_f = x.view(x.size(0), -1)
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return x

class Meta_net(nn.module):
    def __init__(self, output_shape):
        self.output_shape =output_shape
        size= output_shape.view(-1)
        self.fc11 = nn.Linear(2, 32)
        self.fc12 = nn.Linear(32, int(size))

    def forward(self, x):
        fc11_out = F.relu(self.fc11(x))
        conv1_weight = self.fc12(fc11_out).view(self.output_shape)
        masked_weights= torch.zeros(self.output_shape)
        # only some weights is preserved, other's are masked
        masked_weights[:x[0],:x[1], :,:] = conv1_weight[:x[0], :x[1], :, :]

        return masked_weights



def resnet20(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(Bottleneck, [2, 2, 2], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model

def resnet29(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(Bottleneck, [3, 3, 3], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model