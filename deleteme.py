import numpy as np


y = np.zeros(10)
y[1] = 1
print(y.shape)
print(y)
y = np.zeros([10, 1])
y[1] = 1
print(y.shape)
print(y)
