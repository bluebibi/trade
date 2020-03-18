import warnings

import boto3
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from torch import optim

from codes.rl.upbit_rl_constants import GAMMA, LEARNING_RATE, TRAIN_BATCH_SIZE_PERCENT, TRAIN_REPEATS, \
    BUYER_MODEL_SAVE_PATH, \
    SELLER_MODEL_SAVE_PATH, BUYER_MODEL_FILE_NAME, S3_BUCKET_NAME, SELLER_MODEL_FILE_NAME
from common.global_variables import WINDOW_SIZE

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

s3 = boto3.client('s3')

class DeepBuyerPolicy:
    def __init__(self):
        self.q = QNet()
        self.q_target = QNet()
        self.load_model()

        self.q_target.load_state_dict(self.q.state_dict())

        self.buyer_memory = ReplayBuffer(buffer_capacity=100000)
        self.pending_buyer_transition = None
        self.optimizer = optim.Adam(self.q.parameters(), lr=LEARNING_RATE)

    def sample_action(self, observation, info_dic, epsilon):
        action, from_model = self.q.sample_action(observation, epsilon)
        if action:  # 1
            return BuyerAction.MARKET_BUY, from_model
        else:
            return BuyerAction.BUY_HOLD, from_model

    def qnet_copy_to_target_qnet(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save_model(self):
        torch.save(self.q.state_dict(), BUYER_MODEL_SAVE_PATH.format(WINDOW_SIZE))
        s3.upload_file(
            BUYER_MODEL_SAVE_PATH.format(WINDOW_SIZE), S3_BUCKET_NAME,
            "REINFORCEMENT_LEARNING/{0}".format(BUYER_MODEL_FILE_NAME.format(WINDOW_SIZE))
        )

    def load_model(self):
        s3.download_file(
            S3_BUCKET_NAME,
            "REINFORCEMENT_LEARNING/{0}".format(BUYER_MODEL_FILE_NAME.format(WINDOW_SIZE)),
            BUYER_MODEL_SAVE_PATH.format(WINDOW_SIZE)
        )
        self.q.load_state_dict(torch.load(BUYER_MODEL_SAVE_PATH.format(WINDOW_SIZE)))
        print("LOADED BY EXISTING BUYER POLICY MODEL!!!\n")

    def train(self):
        loss_lst = []
        for i in range(TRAIN_REPEATS):
            train_batch_size = int(self.buyer_memory.size() * TRAIN_BATCH_SIZE_PERCENT / 100)
            s, a, r, s_prime, done_mask = self.buyer_memory.sample_memory(train_batch_size)
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
        self.load_model()

        self.q_target.load_state_dict(self.q.state_dict())

        self.seller_memory = ReplayBuffer(buffer_capacity=100000)
        self.optimizer = optim.Adam(self.q.parameters(), lr=LEARNING_RATE)

    def sample_action(self, observation, info_dic, epsilon):
        action, from_model = self.q.sample_action(observation, epsilon)
        if action:  # 1
            return SellerAction.MARKET_SELL, from_model
        else:
            return SellerAction.SELL_HOLD, from_model

    def qnet_copy_to_target_qnet(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save_model(self):
        torch.save(self.q.state_dict(), SELLER_MODEL_SAVE_PATH.format(WINDOW_SIZE))
        s3.upload_file(
            SELLER_MODEL_SAVE_PATH.format(WINDOW_SIZE), S3_BUCKET_NAME,
            "REINFORCEMENT_LEARNING/{0}".format(SELLER_MODEL_FILE_NAME.format(WINDOW_SIZE))
        )

    def load_model(self):
        s3.download_file(
            S3_BUCKET_NAME,
            "REINFORCEMENT_LEARNING/{0}".format(SELLER_MODEL_FILE_NAME.format(WINDOW_SIZE)),
            SELLER_MODEL_SAVE_PATH.format(WINDOW_SIZE)
        )
        self.q.load_state_dict(torch.load(SELLER_MODEL_SAVE_PATH.format(WINDOW_SIZE)))
        print("LOADED BY EXISTING SELLER POLICY MODEL!!!\n")

    def train(self):
        loss_lst = []
        for i in range(TRAIN_REPEATS):
            train_batch_size = int(self.seller_memory.size() * TRAIN_BATCH_SIZE_PERCENT / 100)
            s, a, r, s_prime, done_mask = self.seller_memory.sample_memory(train_batch_size)
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
    def __init__(self, bias=True, input_size=21, hidden_size=256, output_size=2, num_layers=3):
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
            bias=bias
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
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
            from_model = 0
            return random.randint(0, 1), from_model
        else:
            from_model = 1
            if not isinstance(x, torch.Tensor):
                x = torch.unsqueeze(torch.Tensor(x), dim=0)
            out = self.forward(x)
            return out.argmax().item(), from_model


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
