# http://neuralnetworksanddeeplearning.com/chap3.html
import plotting as plot
import pandas as pd
import numpy as np
import scipy.signal as sg
import random
import mnist_loader as mnist
import copy

TRACK_VALIDATION_ACCURACY = True


np.random.seed(12345678)
training_accuracy_history = list()
validation_accuracy_history = list()
training_cost_history = list()
validation_cost_history = list()


def logistic(i):
    return 1 / (1 + np.exp(-1 * i))
logistic = np.vectorize(logistic, otypes=[np.float])


def softmax(i):
    exp = np.exp(i)
    denominators = np.sum(exp, axis=0)
    softmax = exp / denominators[None, :]
    # print softmax
    return softmax


def accuracy(output, labels):
    output_values = np.argmax(output, axis=0)
    output_dist = list()
    label_dist = list()
    label_values = np.argmax(labels, axis=0)
    return sum(output_values == label_values) / float(len(label_values))

# input layer = a0
# input -> a1 = w1 * a0 -> a2 = w2 * a1


def initialize_weights(nn):
    w = dict()
    bias = dict()
    for i in xrange(1, nn.index_of_final_layer + 1):
        limit = np.sqrt(
            float(6) / (nn.layers[i - 1].size + 1 + nn.layers[i].size))
        if nn.layers[i].ctype == Layer.C_FULLYCONNECTED:
            w[i] = np.random.uniform(-1 * limit, limit,
                                     (nn.layers[i].size, nn.layers[i - 1].size))
            bias[i] = np.random.uniform(-1 * limit, limit, (nn.layers[i].size,1))
        elif nn.layers[i].ctype == Layer.C_CONV:
            assert is_square(nn.layers[i].size)
            edge = np.sqrt(nn.layers[i].size)
            w[i] = np.random.uniform(-1 * limit, limit, (edge,edge))
            bias[i] = np.random.uniform(-1 * limit, limit, nn.layers[i].size)
    nn.w = w
    nn.bias = bias


class QuadraticCost():

    @staticmethod
    def cost(outputs, labels):
        return 0.5 * np.sum(np.square(outputs - labels)) / outputs.shape[1]

    @staticmethod
    def delta(outputs, labels, sigma_derivative):
        return np.multiply(outputs - labels, sigma_derivative)


class CrossEntropyCost():

    @staticmethod
    def cost(outputs, labels):
        return np.sum(np.nan_to_num(-labels * np.log(outputs) -
                                    (1 - labels) * np.log(1 - outputs)))

    @staticmethod
    def delta(outputs, labels, sigma_derivative):
        return (outputs - labels)


def track_accuracy(nn, training_data, validation_data, costfunction):

    validation_accuracy = 0
    if TRACK_VALIDATION_ACCURACY:
        training_predictions = feedforward(
            training_data.features, nn)[
            nn.index_of_final_layer]
        training_accuracy = accuracy(
            training_predictions, training_data.labels)
        training_accuracy_history.append(training_accuracy)
        training_cost_history.append(
            costfunction.cost(
                training_predictions, training_data.labels))

    if nn.max_declines_before_stop > 0 or TRACK_VALIDATION_ACCURACY:
        validation_predictions = feedforward(
            validation_data.features, nn)[
            nn.index_of_final_layer]
        validation_accuracy = accuracy(
            validation_predictions,
            validation_data.labels)
        validation_accuracy_history.append(validation_accuracy)
        validation_cost_history.append(
            costfunction.cost(
                validation_predictions, validation_data.labels))
        print "training accuracy: " + str(training_accuracy)
    return validation_accuracy


def train(
        training_data, nn,
        epochs,
        mini_batch_size,
        costfunction=CrossEntropyCost, validation_data=None):
    num_training_data = training_data.labels[1, :].size
    a = dict()
    delta_w = dict()
    delta_bias = dict()
    delta = dict()
    validation_accuracy = -1
    validation_accuracy_decreases = 0
    initialize_weights(nn)
    velocity = dict()
    for index in nn.w:
        velocity[index] = np.zeros_like(nn.w[index])

    for j in range(epochs):  # number of training loops
        print "Starting epoch " + str(j) + " of " + str(epochs)
        if nn.max_declines_before_stop > 0:
            old_validation_accuracy = validation_accuracy

        validation_accuracy = track_accuracy(
            nn, training_data, validation_data, costfunction)

        if nn.max_declines_before_stop > 0:
            if validation_accuracy <= old_validation_accuracy:
                validation_accuracy_decreases += 1
            else:
                validation_accuracy_decreases = 0
            if validation_accuracy_decreases > nn.max_declines_before_stop:
                break

        for mini_batch_index in xrange(0, num_training_data, mini_batch_size):
            old_velocity = copy.deepcopy(velocity)
            features_batch = training_data.features[
                :, mini_batch_index: mini_batch_index + mini_batch_size]
            labels_batch = training_data.labels[
                :, mini_batch_index: mini_batch_index + mini_batch_size]
            this_batch_size = labels_batch[1, :].size
            a = feedforward(features_batch, nn)
# Iterate through the layers
            for l in sorted(nn.w.keys(), reverse=True):
                # compute error, elementwise computations
                sigma_derivative = np.multiply(a[l], 1 - a[l])  # 10 x 1593

                if l == nn.index_of_final_layer:
                    delta[l] = costfunction.delta(
                        a[l], labels_batch, sigma_derivative)

                else:
                    delta[l] = np.multiply(
                        np.dot(
                            nn.w[
                                l + 1].transpose(),
                            delta[
                                l + 1]),
                        sigma_derivative)

             # compute delta W
                bias_deltas = np.zeros(
                    (this_batch_size, nn.bias[l].shape[0], nn.bias[l].shape[1]))
                weight_deltas = np.zeros(
                    (this_batch_size, nn.w[l].shape[0], nn.w[l].shape[1]))
                for datum in range(this_batch_size):
                    delta_l = np.mat(delta[l])[:, datum]
                    a_prev = np.mat(a[l - 1])[:, datum]
                    weight_deltas[datum, :, :], bias_deltas[datum,:,:] = nn.layers[
                        l].delta_w(a_prev, delta_l)#figure out how to get bias_deltas
                delta_w[l] = weight_deltas.mean(axis=0)
                delta_bias[l] = bias_deltas.mean(axis=0)
# Apply regularization
                regularization = 1 - nn.learning_rate * nn.l2_lamda / this_batch_size
                velocity[l] = old_velocity[l] * \
                    nn.friction - nn.learning_rate * delta_w[l]
                nn.w[l] = nn.w[l] * regularization + velocity[l]
                nn.bias[l] = nn.bias[l] - nn.learning_rate * delta_bias[l]


def is_square(n):
    return int(np.sqrt(n))**2 == n


def feedforward(features, nn):
    a = dict()
    a[0] = features
    for l in nn.w:

        z = nn.layers[l].forward_pass(nn.w[l], nn.bias[l], a[l - 1])#.reshape((nn.layers[l].size+1,-1))
        no_bias_a = nn.layers[l].activation_function(z)

        a[l] = no_bias_a
    return a


def test(data, nn):
    a = feedforward(data.features, nn)
    output = a[nn.index_of_final_layer]  # 10 x 1593
    test_accuracy = accuracy(output, data.labels)
    print "test accuracy = " + str(test_accuracy)
    return test_accuracy


class NN():

    def __init__(self, layers):
        self.layers = layers
        self.index_of_final_layer = len(layers) - 1
        self.w = dict()
        self.bias = dict()

# Hyperparameters
        self.max_declines_before_stop = 0
        self.learning_rate = .01
        self.l2_lamda = 0.05
        self.friction = 0.0


class Layer():
    T_LOGISTIC = 'LOGISTIC'
    T_SOFTMAX = 'SOFTMAX'
    C_CONV = 'CONV'
    C_FULLYCONNECTED = 'FULLYCONNECTED'

    def __init__(self, size, ltype=T_LOGISTIC, ctype=C_FULLYCONNECTED):
        self.size = size
        self.ltype = ltype
        self.ctype = ctype
        if ltype == Layer.T_LOGISTIC:
            self.activation_fn = logistic
        else:
            self.activation_fn = softmax

    def activation_function(self, z):
        return self.activation_fn(z)

    def forward_pass(self, w, bias, a_prev):
        if (self.ctype == Layer.C_CONV):

            print w.shape
            print a_prev[:,0].size
            print w.size
            assert is_square(a_prev[:,0].size)
            assert is_square(w.size)
            a_edge_size = np.sqrt(a_prev[:,0].size)
            print a_edge_size
            a_square = np.array(a_prev.reshape((-1,a_edge_size, a_edge_size)))
            print a_square.shape
            w_square = w.reshape((1,w.shape[0], w.shape[1]))
            print w.shape
            z = sg.convolve(a_square, w_square, mode='valid') + bias
            return z
        elif self.ctype == Layer.C_FULLYCONNECTED:
            return np.dot(w, a_prev)+bias
        else:
            raise

    def delta_w(self, a_prev, delta_a):
        if (self.ctype == Layer.C_CONV):
            assert is_square(a_prev[1:].size)
            assert is_square(delta_a.size)
            a_prev = a_prev[1:]  # remove bias
            a_edge_size = np.sqrt(a_prev.size)
            delta_edge_size = np.sqrt(delta_a.size)
            a_square = np.array(a_prev.reshape((a_edge_size, a_edge_size)))
            delta_square = w[1:].reshape((delta_edge_size, delta_edge_size))
            tmp = sg.convolve2d(a_square, delta_square, mode='valid')
            delta_w = np.ones(tmp.size + 1)
            delta_w[1:] = tmp
            delta_w[1] = delta_a.mean()
            return delta_w
        elif self.ctype == Layer.C_FULLYCONNECTED:
            return np.dot(delta_a, a_prev.transpose()), np.dot(delta_a, np.ones((1,a_prev.shape[1])).transpose())
        else:
            raise


class MLDataSet():

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


def hyperparam_search(
        x_min,
        x_max,
        y_min,
        y_max,
        nn,
        training_data,
        validation_data):
    granularity = 2
    eps = 1e-10
    x_range = np.logspace(
        np.log10(
            x_min + eps),
        np.log10(x_max),
        granularity + 1)
    y_range = np.logspace(
        np.log10(
            y_min + eps),
        np.log10(y_max),
        granularity + 1)
    results = np.zeros([granularity, granularity])
    for x_index, x in enumerate(x_range[:-1]):
        for y_index, y in enumerate(y_range[:-1]):
            print "x is " + str(x) + " and y is " + str(y)

            train(training_data, nn, int(x), int(y),
                  validation_data=validation_data)

            print "Indices: " + str(x_index) + ", " + str(y_index)
            results[x_index][y_index] = float(test(validation_data, nn))
    plot.plot_hyperparam_search(
        x_range,
        y_range,
        results.transpose(),
        'epochs',
        'batch size')


def main():
    nn = NN([Layer(784), Layer(25, ctype=Layer.C_FULLYCONNECTED), Layer(10, ltype=Layer.T_LOGISTIC)])
# read in data
    """
    raw_data = pd.read_csv(
        "/Users/delbalso/projects/nn1/data/handwriting.csv",
        sep=",",
        header=None)
    raw_data = raw_data.reindex(np.random.permutation(raw_data.index))
    data = np.array(raw_data.transpose())
    num_labels = 10
    num_test_data = int(data.shape[1] * 0.2)
    features = data[:-num_labels, :]  # num_features x num_examples
    labels = data[-1 * num_labels:, :]  # num_labels x num_examples
    weights = train(
        labels[
            :,
            :-
            1 *
            num_test_data],
        features[
            :,
            :-
            num_test_data],
        nn)
    test(labels[:, -num_test_data:], features[:, -num_test_data:], weights, nn)
    """
    training_data, validation_data, test_data = mnist.load_data_wrapper_1()
    random.shuffle(training_data)
    training_features, training_labels = zip(*training_data[:500])
    training_data = MLDataSet(
        np.squeeze(training_features).transpose(),
        np.squeeze(training_labels).transpose())
    validation_features, validation_labels = zip(*validation_data[:])
    validation_data = MLDataSet(
        np.squeeze(validation_features).transpose(),
        np.squeeze(validation_labels).transpose())
    test_features, test_labels = zip(*test_data)
    test_data = MLDataSet(
        np.squeeze(test_features).transpose(),
        np.squeeze(test_labels).transpose())

    #hyperparam_search(1, 10, 1, 100, nn, training_data, validation_data)
    train(training_data, nn, 30, 10, validation_data=validation_data)

    test(test_data, nn)
    plot.plot_history(
        training_accuracy_history,
        validation_accuracy_history,
        training_cost_history,
        validation_cost_history)


if __name__ == "__main__":
    main()
