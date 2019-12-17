# https://jaeyung1001.tistory.com/45
import numpy as np
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.tests.conftest import F
from common.global_variables import *
from codes.upbit.upbit_order_book_based_data import UpbitOrderBookBasedData
import warnings
warnings.filterwarnings("ignore")


class LSTM(nn.Module):
    def __init__(self, bias=True, dropout=0.25, input_size=63, hidden_size=256, output_size=1, num_layers=3):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=bias,
            dropout=dropout
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(input=x)
        out = self.fc_layer(out)
        out = out[:, -1, :]
        out = F.sigmoid(out)
        return out

    # def init_hidden(self, x):
    #     hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
    #     cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
    #     return hidden, cell


if __name__ == "__main__":
    upbit_order_book_data = UpbitOrderBookBasedData("ADA")
    x_normalized_original, y_up_original, one_rate, total_size = upbit_order_book_data.get_dataset(split=False)

    print(x_normalized_original.shape, y_up_original.shape, one_rate, total_size)

    for y in y_up_original:
        print(y, end=",")
    print()

    lstm_model = LSTM(input_size=63, bias=True, dropout=0.5).to(DEVICE)
    net = NeuralNetClassifier(
        lstm_model,
        max_epochs=10,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adam,
        device=DEVICE,
        criterion=torch.nn.BCELoss,
        lr=0.05
    )

    X = torch.tensor(x_normalized_original, dtype=torch.float)
    y = torch.tensor(y_up_original, dtype=torch.float)

    # print(net.get_params().keys())
    #
    # param_grid = {
    #     'lr': [0.01, 0.05, 0.1],
    #     'module__bias': [True, False],
    #     'module__dropout': [0.0, 0.25, 0.5]
    # }
    #
    # gs = GridSearchCV(net, param_grid, refit=True, cv=5, scoring='f1')
    # gs.fit(X=X, y=y)

    net.fit(X=X, y=y)