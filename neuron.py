import numpy as np


class Neuron(object):

    def activationfn(self, i):
        return None


class Logistic(Neuron):

    def logistic(self, i):
        return 1 / (1 + np.exp(-1 * i))

    def __init__(self):
        self.activationfn = np.vectorize(self.logistic, otypes=[np.float])


class SoftMax(Neuron):

    def activationfn(self, i):
        exp = np.exp(i)
        denominators = np.sum(exp, axis=0)
        softmax = exp / denominators[None, :]
        return softmax


class Linear(Neuron):

    def activationfn(self, i):
        return i
