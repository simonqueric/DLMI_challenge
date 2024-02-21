import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        #out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ConvolutionalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, num_classes, K, pooling="mean"):
        super(ConvolutionalFeatureExtractor, self).__init__()
        self.pooling = pooling
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = self._make_layer(64, K, 1)
        self.conv3 = self._make_layer(K, 2*K, 1)
        self.conv4 = self._make_layer(2*K, 4*K, 1)
        self.conv5 = self._make_layer(4*K, 8*K, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 8 * K, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        if self.pooling=="mean":
            x = x.mean(axis=0)
        if self.pooling=="max":
            x = x.max(axis=0)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, K=2, pooling="mean"):
        super(CNN, self).__init__()
        self.pooling = pooling
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, K, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(K)
        self.conv3 = nn.Conv2d(K, 2*K, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(2*K)
        self.conv4 = nn.Conv2d(2*K, 4*K, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(4*K)
        self.conv5 = nn.Conv2d(4*K, 8*K, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(8*K)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 8 * K, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        if self.pooling=="mean":
            x = x.mean(axis=0)
        if self.pooling=="max":
            x = x.max(axis=0)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x