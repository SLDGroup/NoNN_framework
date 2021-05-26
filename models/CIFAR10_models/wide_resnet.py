import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout, num_classes, mode="train"):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.mode = mode

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        layer_out = F.avg_pool2d(out, 8)
        out = layer_out.view(layer_out.size(0), -1)
        out = self.linear(out)

        if self.mode == "train":
            return out, layer_out
        elif self.mode == "deploy":
            return out
        else:
            print("ERROR: Wrong model mode. Choose form [train, deploy]")
            exit(-1)


if __name__ == '__main__':
    # net = Wide_ResNet(28, 10, 0.3, 10)
    # y = net(Variable(torch.randn(1, 3, 32, 32)))
    # net = Wide_ResNet(16, 1, 0, 10)
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"parameters WRN161= {n_parameters}")
    # net = Wide_ResNet(16, 2, 0, 10)
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"parameters WRN162= {n_parameters}")
    net = Wide_ResNet(40, 4, 0, 10)
    print(net)
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"parameters WRN404= {n_parameters}")

    # Wide-Resnet 16x1
    # parameters WRN161= 175 626
    # Wide-Resnet 16x2
    # parameters WRN162= 692 810
    # Wide-Resnet 40x4
    # parameters WRN404= 8 955 050
