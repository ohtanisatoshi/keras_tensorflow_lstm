import numpy as np

rng = np.random.RandomState(123)

d = 2
N = 10
mean = 5

x1 = rng.randn(N, d) + np.array([0, 0])
x2 = rng.randn(N, d) + np.array([mean, mean])

x = np.concatenate((x1, x2), axis=0)

w = np.zeros(d)
b = 0

def y(x):
    return step(np.dot(x, w) + b)

def step(x):
    return 1 * (x > 0)