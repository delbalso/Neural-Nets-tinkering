import unittest
import nn
import numpy as np

class TestLayer(unittest.TestCase):

    def setUp(self):
        pass

    def test_forward_pass_conv(self):
        layer = nn.Layer(2,ltype=nn.Layer.T_LOGISTIC, ctype = nn.Layer.C_CONV)
        w = np.array([1,1,0,0,1])
        a_prev = np.array([1,1,2,3,4,5,6,7,8,9])
        forward_pass = layer.forward_pass(w, a_prev)
        self.assertTrue((forward_pass==np.array([[7,9],[13,15]])).all())

    def test_forward_pass_fully_connected(self):
        layer = nn.Layer(2,ltype=nn.Layer.T_LOGISTIC, ctype = nn.Layer.C_FULLYCONNECTED)
        w = np.array([[1,0,4,1],[0,0,0,1]])
        a_prev = np.array([1,1,2,9])
        forward_pass = layer.forward_pass(w, a_prev)
        print "forward"
        print forward_pass
        self.assertTrue((forward_pass==np.array([18,9])).all())

    def test_two_layer(self):
        pass

if __name__ == '__main__':
    unittest.main()

