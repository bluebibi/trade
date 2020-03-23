import collections
import random

buffer = collections.deque(maxlen=1000)

buffer.append({1: []})
buffer.append({2: []})
buffer.append({3: []})

mini_batch = random.sample(buffer, 2)

for element in buffer:
    print(element.keys())