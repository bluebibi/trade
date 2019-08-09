import math

import torch.nn as nn
from torch.nn import init


def get_conv2d_size(w, h, kernel_size, padding_size, stride):
    return math.floor((w - kernel_size + 2 * padding_size) / stride) + 1, math.floor((h - kernel_size + 2 * padding_size) / stride) + 1


def get_pool2d_size(w, h, kernel_size, stride):
    return math.floor((w - kernel_size) / stride) + 1, math.floor((h - kernel_size) / stride) + 1


class CNN(nn.Module):
    def __init__(self, input_width, input_height):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3),   # [batch_size,1,28,28] -> [batch_size,16,24,24]
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3),  # [batch_size,16,24,24] -> [batch_size,32,20,20]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),                      # [batch_size,32,20,20] -> [batch_size,32,10,10]
            nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3),  # [batch_size,32,10,10] -> [batch_size,64,6,6]
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)                       # [batch_size,64,6,6] -> [batch_size,64,3,3]
        )

        w, h = get_conv2d_size(w=input_width, h=input_height, kernel_size=3, padding_size=0, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=3, padding_size=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=3, padding_size=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)

        self.fc_layer = nn.Sequential(
            nn.Linear(w * h * 6, 128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size(0), -1)
        out = self.fc_layer(out)
        return out.squeeze(dim=1)


