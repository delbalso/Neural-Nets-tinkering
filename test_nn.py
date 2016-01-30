import unittest
import nn
import neuron
import layers
import numpy as np


class TestLayer(unittest.TestCase):

    def setUp(self):
        self.features = np.zeros((3, 3))
        self.features[:, 0] = [1, 2, 3]
        self.features[:, 1] = [4, 5, 6]
        self.features[:, 2] = [7, 8, 9]

    def test_feed_forward_3_fully_connected(self):
        w = dict()
        bias = dict()
        w[1] = np.array([[1, 2, 3], [40, 50, 60]])
        bias[1] = np.zeros((2, 1))
        layer_1 = layers.FullyConnectedLayer(
            2, 3, list(), unit=neuron.Linear())
        w[2] = np.array([[0, 1], [1, 0]])
        bias[2] = np.zeros((2, 1))
        layer_2 = layers.FullyConnectedLayer(
            2, 2, list(), unit=neuron.Linear())
        w[3] = np.array([[1, 1], [2, 2], [3, 3]])
        bias[3] = np.ones((3, 1))
        layer_3 = layers.FullyConnectedLayer(
            3, 2, list(), unit=neuron.Linear())
        net = nn.NN([None, layer_1, layer_2, layer_3])

        net.w = w
        net.bias = bias
        a = nn.feedforward(self.features, net)

        print "a0"
        print a[0]
        print a[0][:, 0]
        print "a1"
        print a[1]
        print "a2"
        print a[2]
        print "a3"
        print a[3]
# layer 1, data 1
        self.assertTrue(
            (a[1][:, 0] == np.array([[14, 320]])).all())
# layer 1, data 2
        self.assertTrue(
            (a[1][:, 1] == np.array([[32, 770]])).all())
# layer 1, data 3
        self.assertTrue(
            (a[1][:, 2] == np.array([[50, 1220]])).all())

# layer 3, data 1
        self.assertTrue(
            (a[2][:, 0] == np.array([[320, 14]])).all())
# layer 3, data 2
        self.assertTrue(
            (a[2][:, 1] == np.array([[770, 32]])).all())
# layer 3, data 3
        self.assertTrue(
            (a[2][:, 2] == np.array([[1220, 50]])).all())

# layer 3, data 1
        self.assertTrue(
            (a[3][:, 0] == np.array([[335, 669, 1003]])).all())
# layer 3, data 2
        self.assertTrue(
            (a[3][:, 1] == np.array([[803, 1605, 2407]])).all())
# layer 3, data 3
        self.assertTrue((a[3][:, 2] == np.array(
            [[1271, 1270 * 2 + 1, 1270 * 3 + 1]])).all())
if __name__ == '__main__':
    unittest.main()
