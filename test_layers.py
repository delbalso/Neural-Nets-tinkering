import unittest
import layers
import numpy as np

class TestLayer(unittest.TestCase):

    def setUp(self):
        pass

    def test_forward_pass_conv_square_input(self):
        layer = layers.ConvolutionalLayer(2,2,3,3,1,list())
        w = np.array([[1,0],[0,1]])
        bias = 1
        a_prev=np.zeros((3,3,2))
        a_prev[:,:,0]=[[1,2,3],[4,5,6],[7,8,9]]
        a_prev[:,:,1]=[[10,2,3],[40,5,6],[70,8,9]]
        forward_pass = layer.forward_pass(w, bias, a_prev)
        self.assertTrue((forward_pass[:,0]==np.array([[7,9,13,15]])).all())
        self.assertTrue((forward_pass[:,1]==np.array([[16,9,49,15]])).all())

    def test_forward_pass_conv_linear_input(self):
        layer = layers.ConvolutionalLayer(2,2,3,3,1,list())
        w = np.array([[1,0],[0,1]])
        bias = 1
        a_prev=np.zeros((3,3,2))
        a_prev[:,:,0]=[[1,2,3],[4,5,6],[7,8,9]]
        a_prev[:,:,1]=[[10,2,3],[40,5,6],[70,8,9]]
        forward_pass = layer.forward_pass(w, bias, a_prev.reshape((9,2)))
        self.assertTrue((forward_pass[:,0]==np.array([7,9,13,15])).all())
        self.assertTrue((forward_pass[:,1]==np.array([16,9,49,15])).all())

    def test_forward_pass_fully_connected(self):
        layer = layers.FullyConnectedLayer(2,6,list())
        w = np.array([[0,4,1,2,3,4],[1,2,3,4,5,6]])

        bias = np.array([[1],[0]])
        a_prev=np.zeros((6,2))
        a_prev[:,0]=[1,2,4,5,7,8]
        a_prev[:,1]=[10,2,0,5,70,8]
        forward_pass = layer.forward_pass(w, bias,a_prev)
        print "forward"
        print forward_pass[:,0]
        self.assertTrue((forward_pass[:,0]==np.array([76,120])).all())

        print forward_pass[:,1]
        self.assertTrue((forward_pass[:,1]==np.array([261,432])).all())

if __name__ == '__main__':
    unittest.main()

