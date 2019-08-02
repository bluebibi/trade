import math

import torch.nn as nn
from torch.nn import init
from common.global_variables import *


def get_conv2d_size(w, h, kernel_size, padding_size, stride):
    return math.floor((w - kernel_size + 2 * padding_size) / stride) + 1, math.floor((h - kernel_size + 2 * padding_size) / stride) + 1


def get_pool2d_size(w, h, kernel_size, stride):
    return math.floor((w - kernel_size) / stride) + 1, math.floor((h - kernel_size) / stride) + 1


class CNN(nn.Module):
    def __init__(self, input_width, input_height):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2),   # [batch_size,1,28,28] -> [batch_size,16,24,24]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),  # [batch_size,16,24,24] -> [batch_size,32,20,20]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),                      # [batch_size,32,20,20] -> [batch_size,32,10,10]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),  # [batch_size,32,10,10] -> [batch_size,64,6,6]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)                       # [batch_size,64,6,6] -> [batch_size,64,3,3]
        )

        w, h = get_conv2d_size(w=input_width, h=input_height, kernel_size=2, padding_size=0, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=2, padding_size=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=2, padding_size=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)

        self.fc_layer = nn.Sequential(
            nn.Linear(w * h * 64, 200),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size(0), -1)
        out = self.fc_layer(out)
        return out.squeeze(dim=1)


