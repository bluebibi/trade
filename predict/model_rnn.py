import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.tests.conftest import F
from torch.nn import init
from common.global_variables import *
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData


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
            nn.Linear(32, 2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out, _ = self.lstm(input=x)
        out = self.fc_layer(out)
        out = out[:, -1, :]
        out = F.softmax(out, dim=-1)

        print(out)
        return out

    # def init_hidden(self, x):
    #     hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
    #     cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
    #     return hidden, cell


if __name__ == "__main__":
    upbit_order_book_data = UpbitOrderBookBasedData("ADA")
    x_normalized_original, y_up_original, one_rate, total_size = upbit_order_book_data.get_dataset(split=False)
    lstm_model = LSTM(input_size=INPUT_SIZE).to(DEVICE)

    net = NeuralNetClassifier(
        lstm_model,
        max_epochs=10,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    X = x_normalized_original
    y = y_up_original.type(torch.LongTensor)
    print(y)

    print(type(X), type(y))
    print(X.shape, y.shape)
    #    print(X.size(), y.size())
    #    print(X.dim(), y.dim())

    net.fit(X=X, y=y)