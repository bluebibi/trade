import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from torch import optim

from codes.rl.upbit_rl_constants import GAMMA, LEARNING_RATE, TRAIN_BATCH_SIZE, TRAIN_REPEATS

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import random
import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

from codes.rl.upbit_rl_utils import BuyerAction, SellerAction
import torch.nn as nn
import torch
import collections
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

class DeepBuyerPolicy:
    def __init__(self):
        self.q = QNet()
        self.q_target = QNet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.buyer_memory = ReplayBuffer(buffer_capacity=100000)
        self.pending_buyer_transition = None
        self.optimizer = optim.Adam(self.q.parameters(), lr=LEARNING_RATE)

    def sample_action(self, observation, info_dic, epsilon):
        if self.q.sample_action(observation, epsilon):  # 1
            return BuyerAction.MARKET_BUY
        else:
            return BuyerAction.BUY_HOLD

    def qnet_copy_to_target_qnet(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def train(self):
        loss_lst = []
        for i in range(TRAIN_REPEATS):
            s, a, r, s_prime, done_mask = self.buyer_memory.sample_memory(TRAIN_BATCH_SIZE)
            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + GAMMA * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            loss_lst.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = np.average(loss_lst)
        print("*** Buyer Policy Trained (Loss: {0}) ***\n".format(avg_loss))
        return avg_loss


class DeepSellerPolicy:
    def __init__(self):
        self.q = QNet()
        self.q_target = QNet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.seller_memory = ReplayBuffer(buffer_capacity=100000)
        self.optimizer = optim.Adam(self.q.parameters(), lr=LEARNING_RATE)

    def sample_action(self, observation, info_dic, epsilon):
        if self.q.sample_action(observation, epsilon):  # 1
            return SellerAction.MARKET_SELL
        else:
            return SellerAction.SELL_HOLD

    def qnet_copy_to_target_qnet(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def train(self):
        loss_lst = []
        for i in range(TRAIN_REPEATS):
            s, a, r, s_prime, done_mask = self.seller_memory.sample_memory(TRAIN_BATCH_SIZE)
            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + GAMMA * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            loss_lst.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = np.average(loss_lst)
        print("*** Seller Policy Trained (Loss: {0}) ***\n".format(avg_loss))
        return avg_loss


class QNet(nn.Module):
    def __init__(self, bias=True, dropout=0.0, input_size=21, hidden_size=256, output_size=2, num_layers=1):
        super(QNet, self).__init__()
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

    def sample_action(self, x, epsilon):
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.unsqueeze(torch.Tensor(x), dim=0)
            out = self.forward(x)
            return out.argmax().item()


class ReplayBuffer:
    def __init__(self, buffer_capacity):
        self.buffer = collections.deque(maxlen=buffer_capacity)

    def put(self, transition):
        self.buffer.append(transition)

    def sample_memory(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_batch.append(s)
            a_batch.append([a])
            r_batch.append([r])
            s_prime_batch.append(s_prime)
            done_mask_batch.append([done_mask])

        return torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch), \
               torch.tensor(r_batch), torch.tensor(s_prime_batch, dtype=torch.float), \
               torch.tensor(done_mask_batch)

    def size(self):
        return len(self.buffer)