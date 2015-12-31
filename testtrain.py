import unittest
import train
import numpy as np

class TestFeedForward(unittest.TestCase):

    def setUp(self):
        self.input1 = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6],[7,8,9]])
        self.input2 = np.array([[1,2,3],[4,5,6]]).transpose()
        self.input3 = np.array([[1,0],[0,1]]).transpose()
        self.w = dict()

    def test_single_layer(self):
        self.w[1] = np.array([[0,1,0],[0,0,1]])
        nn = train.NN([2,2])
        a = train.feedforward(self.input3, self.w, nn)
        classification = np.array(np.argmax(a[1],axis=0))
        self.assertTrue((classification==np.array([0,1])).all())

    def test_two_layer(self):
        self.w[1] = np.array([[1,1,1,0],[1,2,0,1]])
        self.w[2] = np.array([[0,1,0],[0,0,1]])
        nn = train.NN([3,2,2])
        a = train.feedforward(self.input2, self.w, nn)
        classification = np.array(np.argmax(a[1],axis=0))
        self.assertTrue((abs(a[2]-np.array([[ 0.72750761,  0.73104965],[ 0.73057215,  0.73105852]]))<0.00000001).all())

if __name__ == '__main__':
    unittest.main()

