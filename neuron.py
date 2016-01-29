import numpy as np

def logistic(i):
    return 1 / (1 + np.exp(-1 * i))
logistic = np.vectorize(logistic, otypes=[np.float])


def softmax(i):
    exp = np.exp(i)
    denominators = np.sum(exp, axis=0)
    softmax = exp / denominators[None, :]
    return softmax
