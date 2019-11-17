import os
import sys
import unittest

import torch

from common.global_variables import INPUT_SIZE, WINDOW_SIZE, DEVICE
from predict.model_cnn import CNN
from predict.model_rnn import LSTM
from upbit.upbit_order_book_based_data import UpbitOrderBookBasedData

idx = os.getcwd().index("trade")
PROJECT_HOME = os.getcwd()[:idx] + "trade"
sys.path.append(PROJECT_HOME)


class TestModels(unittest.TestCase):
    def setUp(self):
        self.upbit_data = UpbitOrderBookBasedData("BTC")
        print()

    def test_cnn_model(self):
        x = self.upbit_data.get_dataset_for_buy()
        model = CNN(input_size=INPUT_SIZE, input_height=WINDOW_SIZE).to(DEVICE)
        model.eval()
        out = model.forward(x)
        out = torch.sigmoid(out)
        t = torch.tensor(0.5).to(DEVICE)
        output_index = (out > t).float() * 1

        prob = out.item()
        idx = int(output_index.item())

        print(prob, idx)

    def test_lstm_model(self):
        x = self.upbit_data.get_dataset_for_buy(model_type="LSTM")
        model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
        model.eval()
        out = model.forward(x)
        out = torch.sigmoid(out)
        t = torch.tensor(0.5).to(DEVICE)
        output_index = (out > t).float() * 1

        prob = out.item()
        idx = int(output_index.item())

        print(prob, idx)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
