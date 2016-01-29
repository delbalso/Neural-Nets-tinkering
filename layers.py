import numpy as np
import scipy.signal as sg
import neuron

def is_square(n):
    return int(np.sqrt(n))**2 == n

class Layer(object):
    T_LOGISTIC = 'LOGISTIC'
    T_SOFTMAX = 'SOFTMAX'
    def __init__(self, size,prev_size,inputs, ltype=T_LOGISTIC):
        self.size = size
        self.ltype = ltype
        self.prev_size = prev_size
        self.inputs = inputs
        assert isinstance(inputs,list)
        self.inputs_size = sum(x.size for x in inputs)
        if ltype == Layer.T_LOGISTIC:
            self.activation_fn = neuron.logistic
        else:
            self.activation_fn = softmax

    def activation_function(self, z):
        return self.activation_fn(z)

    def toVector(self,a):
        return a.reshape((self.size,-1))

class ConvolutionalLayer(Layer):

    def __init__(self, kernel_size_x, kernel_size_y, prev_x, prev_y, num_feature_maps, inputs, ltype=Layer.T_LOGISTIC):
        super(ConvolutionalLayer,self).__init__(None,prev_x * prev_y,inputs)
        self.prev_x = prev_x
        self.prev_y = prev_y
        self.size_x = 1 + prev_x - kernel_size_x
        self.size_y = 1 + prev_y - kernel_size_y
        self.size = self.size_x * self.size_y
        self.ltype = ltype
        self.kernel_size = kernel_size_x, kernel_size_y

    def forward_pass(self, w, bias, a_prev):
        assert is_square(a_prev.size%self.prev_x*self.prev_y==0)
        assert is_square(w.size)
        a_square = np.array(a_prev.reshape((self.prev_x, self.prev_y,-1))) # make it square
        w_square = w.reshape((w.shape[0], w.shape[1],-1)) # make it 3d
        z = sg.convolve(a_square, w_square, mode='valid')
        z = z + int(bias)
        return self.toVector(z)

    def delta_w(self, a_prev, delta_a):
        assert is_square(a_prev.size)
        assert is_square(delta_a.size)
        a_edge_size = np.sqrt(a_prev.size)
        delta_edge_size = np.sqrt(delta_a.size)
        a_square = a_prev.reshape((a_edge_size, a_edge_size))
        delta_square = delta_a.reshape((delta_edge_size, delta_edge_size))
        delta_w = sg.convolve2d(a_square, delta_square, mode='valid')
        delta_bias = delta_a.mean()
        return delta_w, delta_bias

class FullyConnectedLayer(Layer):
    def forward_pass(self, w, bias, a_prev):
        print "passing forward"
        print bias
        assert(bias.ndim==2)
        assert(bias.shape[1]==1)
        print w.shape
        print a_prev.shape
        output = np.dot(w, a_prev)+bias
        print output.shape

        return output

    def delta_w(self, a_prev, delta_a):
        return np.dot(delta_a, a_prev.transpose()), np.dot(delta_a, np.ones((1,a_prev.shape[1])).transpose())

