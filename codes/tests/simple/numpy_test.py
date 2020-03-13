import numpy as np

a = np.ones([10, 3])
a[:, 2] = 2.0
b = a[:, 0] + a[:, 2]

b = np.expand_dims(b, axis=1)
print(a, a.shape)
print(b, b.shape)

c = np.append(a, b, axis=1)
print(c)