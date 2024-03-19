from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from models import MLP
import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, K=64):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=K, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=2*K, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=3*K, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=4*K, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * 4, num_classes)
        self.fc = nn.LazyLinear(num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = x.reshape(x.shape[0], -1)
        x = x.mean(axis=0)
        x = x.flatten()

        x = self.fc(x)

        return self.sigmoid(x)

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet18(img_channel=3, num_classes=1) :
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes)

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


class ModifiedResNet(nn.Module):
    def __init__(self, resnet):
        super(ModifiedResNet, self).__init__()
        self.resnet = resnet
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.LazyLinear(1)
        self.mlp  = MLP()
        # self.fc1 = nn.Linear(512*7**2, 512)
        # self.fc2 = nn.Linear(512, 1)
        # self.relu = nn.ReLU()
    def forward(self, x, y):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        # print(x.shape)
        # x = x.max(axis=0)
        x = x.mean(axis=0)
        # print(x.shape)
        x = torch.flatten(x)
        x = torch.cat((x, y.flatten()), 0)

        x = self.fc(x)
        # y = self.mlp(y).squeeze(0)
        # out = (x+y)/2
        out = self.sigmoid(x)
        return out

def test():
    BATCH_SIZE = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net = ResNet18(img_channel=3, num_classes=1).to(device)
    resnet = models.resnet18(pretrained=True).to(device)
    resnet.fc = nn.Linear(512, 1).to(device)
    net = ResNetWithPooling(resnet).to(device)
    #print(resnet.eval())
    input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
    print(resnet.fc)

    print(net(input))

    # loss_fn = nn.BCELoss()
    # # test
    
    
    # y = net(input).to(device)
    # loss = loss_fn(y, torch.Tensor(1).to(device))
    # loss.backward()
    # print(y)


if __name__ == "__main__":
    test()
