import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(input_size=3, hidden_size=3, batch_first=True)  # Input dim is 3, output dim is 3

print("LSTM:", lstm)

inputs = [torch.zeros(1, 3) for _ in range(8)]  # make a sequence of length 5
for input in inputs:
    input[0][0] = 1.0
    input[0][1] = 2.0
    input[0][2] = 3.0

print("inputs:", inputs)

print()

inputs = torch.cat(inputs).view(2, 4, -1)
hidden = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))  # clean out hidden state

print("inputs.size():", inputs.size())
print("inputs:", inputs)

out, hidden = lstm(inputs, hidden)
print("out", out)
print("hidden", hidden)