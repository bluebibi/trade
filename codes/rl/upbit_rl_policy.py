import math
import warnings

import boto3
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from torch import optim

from codes.rl.upbit_rl_replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

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

from codes.rl.upbit_rl_constants import GAMMA, LEARNING_RATE, TRAIN_BATCH_SIZE_PERCENT, TRAIN_REPEATS, \
    BUYER_MODEL_SAVE_PATH, SELLER_MODEL_SAVE_PATH, BUYER_MODEL_FILE_NAME, S3_BUCKET_NAME, SELLER_MODEL_FILE_NAME, \
    TRAIN_BATCH_MIN_SIZE, REPLAY_MEMORY_SIZE, WINDOW_SIZE, SIZE_OF_FEATURE, SIZE_OF_FEATURE_WITHOUT_VOLUME

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

s3 = boto3.client('s3')


class DeepBuyerPolicy:
    def __init__(self, args=None):
        self.args = args

        if self.args.volume:
            self.input_size = SIZE_OF_FEATURE
        else:
            self.input_size = SIZE_OF_FEATURE_WITHOUT_VOLUME

        if self.args.lstm:
            self.q = QNet_LSTM(input_size=self.input_size)
            self.q_target = QNet_LSTM(input_size=self.input_size)
        else:
            self.q = QNet_CNN(input_size=self.input_size, input_height=WINDOW_SIZE)
            self.q_target = QNet_CNN(input_size=self.input_size, input_height=WINDOW_SIZE)
        self.load_model()

        if self.args.per:
            self.buyer_memory = PrioritizedReplayBuffer(capacity=REPLAY_MEMORY_SIZE)
        else:
            self.buyer_memory = ReplayBuffer(capacity=REPLAY_MEMORY_SIZE)
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
        torch.save(self.q.state_dict(), BUYER_MODEL_SAVE_PATH.format(
            "LSTM" if self.args.lstm else "CNN",
            WINDOW_SIZE,
            SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
        ))
        if self.args.federated:
            s3.upload_file(
                BUYER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                ),
                S3_BUCKET_NAME,
                "REINFORCEMENT_LEARNING/{0}".format(BUYER_MODEL_FILE_NAME.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                ))
            )

    def load_model(self):
        if self.args.federated:
            s3.download_file(
                S3_BUCKET_NAME,
                "REINFORCEMENT_LEARNING/{0}".format(BUYER_MODEL_FILE_NAME.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                )),
                BUYER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                )
            )
            self.q.load_state_dict(torch.load(BUYER_MODEL_SAVE_PATH.format(
                "LSTM" if self.args.lstm else "CNN",
                WINDOW_SIZE,
                SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
            )))
            print("LOADED BY EXISTING BUYER POLICY MODEL FROM AWS S3!!!\n")
        else:
            if os.path.exists(BUYER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
            )):
                self.q.load_state_dict(torch.load(BUYER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                )))
                print("LOADED BY EXISTING BUYER POLICY MODEL FROM LOCAL STORAGE!!!\n")

        self.qnet_copy_to_target_qnet()

    def train(self, beta):
        loss_lst = []
        for i in range(TRAIN_REPEATS):
            train_batch_size = min(
                TRAIN_BATCH_MIN_SIZE,
                int(self.buyer_memory.size() * TRAIN_BATCH_SIZE_PERCENT / 100)
            )

            if self.args.per:
                s, a, r, s_prime, done_mask, indices, weights = self.buyer_memory.sample_priority_memory(
                    train_batch_size, beta=beta
                )
            else:
                s, a, r, s_prime, done_mask = self.buyer_memory.sample_memory(train_batch_size)

            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1).detach()
            target = r + GAMMA * max_q_prime * done_mask

            if self.args.per:
                weights = torch.FloatTensor(weights)
                q_a = torch.squeeze(q_a, dim=1)
                target = torch.squeeze(target, dim=1)
                loss = (q_a - target).pow(2) * weights
                prios = loss + 1e-5
                loss = loss.mean()
                loss_lst.append(loss.item())
                self.buyer_memory.update_priorities(indices, prios.data.cpu().numpy())
            else:
                q_a = torch.squeeze(q_a, dim=1)
                target = torch.squeeze(target, dim=1)
                loss = (q_a - target).pow(2).mean()
                loss_lst.append(loss.item())


            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.q.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        avg_loss = np.average(loss_lst)
        #print("*** Buyer Policy Trained (Loss: {0}) ***\n".format(avg_loss))
        return avg_loss


class DeepSellerPolicy:
    def __init__(self, args=None):
        self.args = args

        if self.args.volume:
            self.input_size = SIZE_OF_FEATURE
        else:
            self.input_size = SIZE_OF_FEATURE_WITHOUT_VOLUME

        if self.args.lstm:
            self.q = QNet_LSTM(input_size=self.input_size)
            self.q_target = QNet_LSTM(input_size=self.input_size)
        else:
            self.q = QNet_CNN(input_size=self.input_size, input_height=WINDOW_SIZE)
            self.q_target = QNet_CNN(input_size=self.input_size, input_height=WINDOW_SIZE)
        self.load_model()

        if self.args.per:
            self.seller_memory = PrioritizedReplayBuffer(capacity=REPLAY_MEMORY_SIZE)
        else:
            self.seller_memory = ReplayBuffer(capacity=REPLAY_MEMORY_SIZE)

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
        torch.save(self.q.state_dict(), SELLER_MODEL_SAVE_PATH.format(
            "LSTM" if self.args.lstm else "CNN",
            WINDOW_SIZE,
            SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
        ))
        if self.args.federated:
            s3.upload_file(
                SELLER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                ),
                S3_BUCKET_NAME,
                "REINFORCEMENT_LEARNING/{0}".format(SELLER_MODEL_FILE_NAME.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                ))
            )

    def load_model(self):
        if self.args.federated:
            s3.download_file(
                S3_BUCKET_NAME,
                "REINFORCEMENT_LEARNING/{0}".format(SELLER_MODEL_FILE_NAME.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                )),
                SELLER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                )
            )
            self.q.load_state_dict(torch.load(SELLER_MODEL_SAVE_PATH.format(
                "LSTM" if self.args.lstm else "CNN",
                WINDOW_SIZE,
                SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
            )))
            print("LOADED BY EXISTING SELLER POLICY MODEL FROM AWS S3!!!\n")
        else:
            if os.path.exists(SELLER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
            )):
                self.q.load_state_dict(torch.load(SELLER_MODEL_SAVE_PATH.format(
                    "LSTM" if self.args.lstm else "CNN",
                    WINDOW_SIZE,
                    SIZE_OF_FEATURE if self.args.volume else SIZE_OF_FEATURE_WITHOUT_VOLUME
                )))
                print("LOADED BY EXISTING SELLER POLICY MODEL FROM LOCAL STORAGE!!!\n")

        self.qnet_copy_to_target_qnet()

    def train(self, beta):
        loss_lst = []
        for i in range(TRAIN_REPEATS):
            train_batch_size = min(
                TRAIN_BATCH_MIN_SIZE,
                int(self.seller_memory.size() * TRAIN_BATCH_SIZE_PERCENT / 100)
            )

            if self.args.per:
                s, a, r, s_prime, done_mask, indices, weights = self.seller_memory.sample_priority_memory(
                    train_batch_size, beta=beta
                )
            else:
                s, a, r, s_prime, done_mask = self.seller_memory.sample_memory(train_batch_size)

            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1).detach()
            target = r + GAMMA * max_q_prime * done_mask

            if self.args.per:
                weights = torch.FloatTensor(weights)
                q_a = torch.squeeze(q_a, dim=1)
                target = torch.squeeze(target, dim=1)
                loss = (q_a - target).pow(2) * weights
                prios = loss + 1e-5
                loss = loss.mean()
                loss_lst.append(loss.item())
                self.seller_memory.update_priorities(indices, prios.data.cpu().numpy())
            else:
                q_a = torch.squeeze(q_a, dim=1)
                target = torch.squeeze(target, dim=1)
                loss = (q_a - target).pow(2).mean()
                loss_lst.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.q.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        avg_loss = np.average(loss_lst)
        #print("*** Seller Policy Trained (Loss: {0}) ***\n".format(avg_loss))
        return avg_loss


class QNet_CNN(nn.Module):
    @staticmethod
    def get_conv2d_size(w, h, kernel_size, padding_size, stride):
        return math.floor((w - kernel_size + 2 * padding_size) / stride) + 1, math.floor(
            (h - kernel_size + 2 * padding_size) / stride) + 1

    @staticmethod
    def get_pool2d_size(w, h, kernel_size, stride):
        return math.floor((w - kernel_size) / stride) + 1, math.floor((h - kernel_size) / stride) + 1

    def __init__(self, input_height, input_size, output_size=2):  #input_size=36, input_height=21
        super(QNet_CNN, self).__init__()
        self.output_size = output_size

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

        w, h = self.get_conv2d_size(w=input_size, h=input_height, kernel_size=3, padding_size=0, stride=1)
        w, h = self.get_conv2d_size(w=w, h=h, kernel_size=3, padding_size=0, stride=1)
        w, h = self.get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)
        w, h = self.get_conv2d_size(w=w, h=h, kernel_size=3, padding_size=0, stride=1)
        w, h = self.get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)

        self.fc_layer = nn.Sequential(
            nn.Linear(w * h * 6, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = self.layer(x)
        out = out.view(x.size(0), -1)
        out = self.fc_layer(out)
        return out.squeeze(dim=1)

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


class QNet_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=2, num_layers=3, bias=True, ):
        super(QNet_LSTM, self).__init__()
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
            nn.Linear(128, self.output_size)
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