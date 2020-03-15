# https://jaeyung1001.tistory.com/45
import numpy as np
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.tests.conftest import F
from common.global_variables import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class LSTM(nn.Module):
    def __init__(self, bias=True, dropout=0.25, input_size=63, hidden_size=256, output_size=2, num_layers=3):
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
            nn.Linear(self.hidden_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, _ = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc_layer(out)
        out = torch.sigmoid(out)
        return out

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


if __name__ == "__main__":
    train_batch_size = 100
    seq_len = 64
    feature_size = 16

    X_train = np.zeros(shape=(train_batch_size, seq_len, feature_size))
    y_up_train = np.zeros(shape=(train_batch_size, 2))

    for idx in range(train_batch_size):
        if idx % 2:
            X_train[idx, :, :] = 0.0
            y_up_train[idx] = [0.0, 1.0]
        else:
            X_train[idx, :, :] = 1.0
            y_up_train[idx] = [1.0, 0.0]

    test_batch_size = 20
    X_test = np.zeros(shape=(test_batch_size, seq_len, feature_size))
    y_up_test = np.zeros(shape=(test_batch_size, 2))

    for idx in range(test_batch_size):
        if idx % 2:
            X_test[idx, :, :] = 0.0
            y_up_test[idx] = [0.0, 1.0]
        else:
            X_test[idx, :, :] = 1.0
            y_up_test[idx] = [1.0, 0.0]

    # print(X_train, y_up_train)
    # print(X_test, y_up_test)

    print(X_train.shape, y_up_train.shape)
    print(X_test.shape, y_up_test.shape)

    X_train = torch.Tensor(X_train)
    y_up_train = torch.Tensor(y_up_train)

    X_test = torch.Tensor(X_test)
    y_up_test = torch.Tensor(y_up_test)

    MAX_EPOCHS = 10
    learning_rate = 0.001

    model = LSTM(input_size=feature_size, bias=True, dropout=0.5).to(DEVICE)
    loss_fn = torch.nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hist = np.zeros(MAX_EPOCHS)

    for t in range(MAX_EPOCHS):
        #model.hidden = model.init_hidden(X_train.size()[0])

        # Forward pass
        y_pred = model(X_train)

        loss = loss_fn(y_pred, y_up_train)
        print("Epoch ", t, "MSE: ", loss.item())

        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    plt.plot(y_pred.detach().numpy(), label="Preds")
    plt.plot(y_up_train.detach().numpy(), label="Data")
    plt.legend()
    plt.show()

    plt.plot(hist, label="Training loss")
    plt.legend()
    plt.show()

    y_test_pred = model(X_test)

    print(y_test_pred)