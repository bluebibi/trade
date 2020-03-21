import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import random
import sys, os

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)

import torch
import collections

class PrioritizedReplayBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def put(self, transition):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample_memory(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        mini_batch = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_batch.append(s)
            a_batch.append([a])
            r_batch.append([r])
            s_prime_batch.append(s_prime)
            done_mask_batch.append([done_mask])

        s_batch_, a_batch_, r_batch_, s_prime_batch_, done_mask_batch_ = \
            torch.tensor(s_batch, dtype=torch.float), \
            torch.tensor(a_batch), \
            torch.tensor(r_batch), \
            torch.tensor(s_prime_batch, dtype=torch.float), \
            torch.tensor(done_mask_batch)

        return s_batch_, a_batch_, r_batch_, s_prime_batch_, done_mask_batch_, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        # print(batch_indices, "!!!", batch_priorities, "!!!")

        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def size(self):
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

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

        try:
            s_batch_, a_batch_, r_batch_, s_prime_batch_, done_mask_batch_ = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch), \
                   torch.tensor(r_batch), torch.tensor(s_prime_batch, dtype=torch.float), \
                   torch.tensor(done_mask_batch)
        except TypeError as e:
            print(e)
            print("s_batch", s_batch)
            print("a_batch", s_batch)
            print("r_batch", s_batch)
            print("s_prime_batch", s_batch)
            print("dpne_mask_batch", s_batch)
            sys.exit(-1)

        return s_batch_, a_batch_, r_batch_, s_prime_batch_, done_mask_batch_

    def size(self):
        return len(self.buffer)