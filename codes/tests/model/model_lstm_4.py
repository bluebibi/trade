import torch
import torch.nn as nn
import numpy as np

EPOCHS = 500
IN_SIZE = 5
NUM_SAMPLES = 5

def generate_data(rows, columns, samples):
	X = []
	y = []
	transformations = {
		'11': lambda x, y: x + y,
		'15': lambda x, y: x - y,
		'10': lambda x, y: x * y,
		'30': lambda x, y: x / y,
		'2': lambda x, y: x + y,
		}
	for j in range(samples):
		data_set = []
		for i in range(columns):
			data = []
			for val, fn in transformations.items():
				data.append(int(fn(int(val), i+j+1)))
			data_set.append(data)
		X.append(data_set)
		y.append([j+1])
	return X, y


class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		self.rnn = nn.LSTM(
			input_size=5,
			hidden_size=NUM_SAMPLES+1,
			num_layers=2,
			batch_first=True,
		)

	def forward(self, x):
		out, (h_n, h_c) = self.rnn(x, None)
		return out[:, -1, :]	# Return output at last time-step


X, y = generate_data(IN_SIZE, 5, NUM_SAMPLES)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

print(X.size())
print(y.size())

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()


for j in range(EPOCHS):
	for i, item in enumerate(X):
		item = item.unsqueeze(0)
		output = rnn(item)
		loss = loss_func(output, y[i])
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if j % 5 == 0:
		print('Loss: ', np.average(loss.detach()))

print('Testing:\n========')
for i, item in enumerate(X):
	print(y[i])
	outp = rnn(item.unsqueeze(0))
	print(np.argmax(outp.detach()))