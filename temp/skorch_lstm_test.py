import numpy as np
import torch
from torch import nn
import skorch

class lstmNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(lstmNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        self.lin = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )

    def forward(self, x):
        out, hn = self.lstm(input=x)
        out = self.lin(out)
        print("!!!", out.shape)
        return out

input_feat = 5
hidden_size = 10
lstmLayers = 2

batch = 30
seq = 20
features = 5

net = skorch.NeuralNet(
    module=lstmNet(
        input_size=input_feat,
        hidden_size=hidden_size,
        num_layers=lstmLayers
    ),
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.SGD,
    lr=0.1,
    max_epochs=10
)

if __name__ == "__main__":
    X = np.random.rand(batch, seq, features)
    X = X.astype(np.float32)
    print('input shape: {}'.format(X.shape))

    y = np.random.rand((batch))
    y = y.astype(np.float32)
    print('target shape: {}'.format(y.shape))

    net.fit(X=X, y=y)