import torch
import torch.nn as nn
from torch.nn import init
from common.global_variables import *


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=3):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            dropout=0.4,
            batch_first=True
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        hidden, cell = self.init_hidden(x)

        out, _ = self.lstm(input=x, hx=(hidden, cell))
        out = self.fc_layer(out[:, -1, :])

        return out.squeeze(dim=1)

    def init_hidden(self, x):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        return hidden, cell
