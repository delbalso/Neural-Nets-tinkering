import unittest
import train
import numpy as np

class TestFeedForward(unittest.TestCase):

    def setUp(self):
        self.input1 = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6],[7,8,9]])
        self.input2 = np.array([[1,2,3],[4,5,6]])
        self.input3 = np.array([[1,0],[0,1]])
        self.w = dict()

    def test_single_layer(self):
        # 2 features, 2 examples
        self.w[1] = np.array([[1,0,0],[0,1,0]])
        nn = train.NN([2,2])

        self.assertEqual(np.argmax(train.feedforward(self.input3, self.w, nn),axis=0),
                         np.argmax(np.array([0,1])))

if __name__ == '__main__':
    unittest.main()

