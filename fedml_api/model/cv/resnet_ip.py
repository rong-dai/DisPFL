'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

__all__ = ['ResNet_ip', 'resnet110']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class per_batch_norm(nn.BatchNorm2d):

    def forward(self, input: Tensor, weight, bias) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight, bias, bn_training, exponential_average_factor, self.eps)


class Bottleneck(nn.Module):
    expansion = 4

    class perBatchNorm2d(nn.BatchNorm2d):
        def forward(self, input: Tensor) -> Tensor:
            self._check_input_dim(input)

            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked = self.num_batches_tracked + 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, per_factor=0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = per_batch_norm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1_g = conv1x1(inplanes, width)
        self.conv1_v = conv1x1(inplanes, width)

        self.bn1_g = norm_layer(width, track_running_stats=False)
        self.bn1_v = norm_layer(width, track_running_stats=False)

        self.conv2_g = conv3x3(width, width, stride, groups, dilation)
        self.conv2_v = conv3x3(width, width, stride, groups, dilation)

        self.bn2_g = norm_layer(width, track_running_stats=False)
        self.bn2_v = norm_layer(width, track_running_stats=False)

        self.conv3_g = conv1x1(width, planes * self.expansion)
        self.conv3_v = conv1x1(width, planes * self.expansion)

        self.bn3_g = norm_layer(planes * self.expansion, track_running_stats=False)
        self.bn3_v = norm_layer(planes * self.expansion, track_running_stats=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.per_factor = per_factor

    def forward(self, x):
        identity = x

        out = self.conv1_g._conv_forward(x, weight= self.conv1_g.weight +  self.conv1_v.weight)
        out = self.bn1_g.forward(out, weight= self.bn1_g.weight + self.bn1_v.weight,
                                 bias= self.bn1_g.bias +  self.bn1_v.bias)

        out = self.relu(out)

        out = self.conv2_g._conv_forward(out, weight=  self.conv2_g.weight  +  self.conv2_v.weight)
        out = self.bn2_g.forward(out, weight=  self.bn2_g.weight + self.bn2_v.weight,
                                 bias= self.bn2_g.bias + self.bn2_v.bias)
        out = self.relu(out)
        out = self.conv3_g._conv_forward(out, weight= self.conv3_g.weight + self.conv3_v.weight)
        out = self.bn3_g.forward(out, weight= self.bn3_g.weight +   self.bn3_v.weight,
                                 bias= self.bn3_g.bias  + self.bn3_v.bias)

        if self.downsample is not None:
            identity = self.downsample.conv1_g._conv_forward(x, weight= self.downsample.conv1_g.weight +  self.downsample.conv1_v.weight)
            identity = self.downsample.bn1_g.forward(identity, weight=  self.downsample.bn1_g.weight +  self.downsample.bn1_v.weight,
                                                     bias=self.downsample.bn1_g.bias + self.downsample.bn1_v.bias)
        out += identity
        out = self.relu(out)

        return out


class ResNet_ip(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, per_factor=0):
        super(ResNet_ip, self).__init__()
        self.per_factor = per_factor
        if norm_layer is None:
            norm_layer = per_batch_norm
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1_g = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.conv1_v = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                 bias=False)

        self.bn1_g = norm_layer(self.inplanes, track_running_stats=False)
        self.bn1_v = norm_layer(self.inplanes, track_running_stats=False)

        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_g = nn.Linear(64 * block.expansion, num_classes)

        self.fc_v = nn.Linear(64 * block.expansion, num_classes)

        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, per_batch_norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3_g.weight, 0)
                    nn.init.constant_(m.bn3_v.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OrderedDict([
                    ('conv1_g', conv1x1(self.inplanes, planes * block.expansion, stride)),
                    ('bn1_g', norm_layer(planes * block.expansion, track_running_stats=False)),
                    ('conv1_v', conv1x1(self.inplanes, planes * block.expansion, stride)),
                    ('bn1_v', norm_layer(planes * block.expansion, track_running_stats=False))
                ])
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, per_factor=self.per_factor))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1_g._conv_forward(x, weight=  self.conv1_g.weight +  self.conv1_v.weight
                                       )

        x = self.bn1_g.forward(x, weight=  self.bn1_g.weight + self.bn1_v.weight,
                               bias= self.bn1_g.bias +  self.bn1_v.bias)

        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = F.linear(x_f, weight= self.fc_g.weight + self.fc_v.weight,
                     bias= self.fc_g.bias +  self.fc_v.bias)

        if self.KD == True:
            return x_f, x
        else:
            return x


def resnet29_ip(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet_ip(Bottleneck, [3, 3, 3], class_num, **kwargs)
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


def resnet56_ip(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet_ip(Bottleneck, [6, 6, 6], class_num, **kwargs)
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


def resnet110_ip(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    logging.info("path = " + str(path))
    model = ResNet_ip(Bottleneck, [12, 12, 12], class_num, **kwargs)
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
