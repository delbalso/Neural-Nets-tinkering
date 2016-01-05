# http://neuralnetworksanddeeplearning.com/chap3.html
import pandas as pd
import numpy as np
import random
import mnist_loader as mnist

np.random.seed(12345678)


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
    for i in xrange(10):
        output_dist.append((output_values == i).sum())
        label_dist.append((label_values == i).sum())
    print output_dist
    print label_dist
    return sum(output_values == label_values) / float(len(label_values))

# input layer = a0
# input -> a1 = w1 * a0 -> a2 = w2 * a1


def train(labels, features, nn, epochs, mini_batch_size):
    # initalize
    num_training_data = labels[1, :].size
    a = dict()
# set up inital weights
    w = dict()
    delta_w = dict()
    delta = dict()
    for i in xrange(1, nn.index_of_final_layer + 1):
        limit = np.sqrt(
            float(6) / (nn.layers[i - 1].size + 1 + nn.layers[i].size))
        w[i] = np.random.uniform(-1 * limit, limit,
                                 (nn.layers[i].size, nn.layers[i - 1].size + 1))

    for j in range(epochs):  # number of training loops
        print "Starting epoch " + str(j) + " of " + str(epochs)

        for mini_batch_index in xrange(0,num_training_data,mini_batch_size):
            features_batch = features[:,mini_batch_index:mini_batch_index+mini_batch_size]
            labels_batch = labels[:,mini_batch_index:mini_batch_index+mini_batch_size]
            this_batch_size = labels_batch[1,:].size
            a = feedforward(features_batch, w, nn)
            if j % 10 == 0 and mini_batch_index==0:
                training_accuracy = accuracy(a[nn.index_of_final_layer], labels_batch)
                if training_accuracy == 1:
                    break
                print "training accuracy: " + str(training_accuracy)
            for l in sorted(w.keys(), reverse=True):
                # compute error, elementwise computations
                sigma_derivative = np.multiply(a[l], 1 - a[l])  # 10 x 1593

                if l == nn.index_of_final_layer:
                    # delta[l] = np.multiply(a[l] - labels_batch, sigma_derivative) # 10
                    # x 1593
                    # cross-entropy loss function (or log loss for softmax)
                    delta[l] = (a[l] - labels_batch)

                elif l == nn.index_of_final_layer - 1:
                    delta[l] = np.multiply(
                        np.dot(w[l + 1].transpose(), delta[l + 1]), sigma_derivative)  # 10 x 1593
                else:
                    delta[l] = np.multiply(
                        np.dot(w[l + 1].transpose(), delta[l + 1][1:]), sigma_derivative)  # 10 x 1593

             # compute delta W
                weight_deltas = np.zeros(
                    (this_batch_size, nn.layers[l].size, nn.layers[l - 1].size + 1))
                learning_rate = .15
                l2_param = 0.1
                for datum in range(this_batch_size):
                    if l == nn.index_of_final_layer:
                        weight_deltas[datum, :, :] = np.dot(np.mat(delta[l])[:, datum], np.mat(
                            a[l - 1])[:, datum].transpose())  # mat multiplication
                    else:
                        weight_deltas[datum, :, :] = np.dot(np.mat(delta[l])[1:, datum], np.mat(
                            a[l - 1])[:, datum].transpose())  # mat multiplication
                delta_w[l] = weight_deltas.mean(axis=0)
# Apply regularization
                regularization = 1 - learning_rate * l2_param / this_batch_size
                regularization_mat = np.append(
                    [1], np.ones(w[l].shape[1] - 1) * regularization)
                w[l] = w[l] * regularization_mat[None, :] - \
                    learning_rate * delta_w[l]
    return w


def feedforward(features, w, nn):
    a = dict()
    a[0] = np.ones((features.shape[0] + 1, features.shape[1]))
    a[0][1:, :] = features
    for l in w:
        z = np.dot(w[l], a[l - 1])
        if (nn.layers[l].ltype == Layer.T_LOGISTIC):
            no_bias_a = logistic(z)
        elif (nn.layers[l].ltype == Layer.T_SOFTMAX):
            no_bias_a = softmax(z)
        else:
            raise
        if l == nn.index_of_final_layer:
            a[l] = no_bias_a
        else:
            a[l] = np.ones((no_bias_a.shape[0] + 1, no_bias_a.shape[1]))
            a[l][1:, :] = no_bias_a
    return a


def test(labels, features, weights, nn):
    a = feedforward(features, weights, nn)
    output = a[nn.index_of_final_layer]  # 10 x 1593
    # print output
    test_accuracy = accuracy(output, labels)
    print "test accuracy = " + str(test_accuracy)


class NN():

    def __init__(self, layers):
        self.layers = layers
        self.index_of_final_layer = len(layers) - 1


class Layer():
    T_LOGISTIC = 'LOGISTIC'
    T_SOFTMAX = 'SOFTMAX'

    def __init__(self, size, ltype=T_LOGISTIC):
        self.size = size
        self.ltype = ltype


def main():
    nn = NN([Layer(784), Layer(100), Layer(10, ltype=Layer.T_SOFTMAX)])
#read in data
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
    training_features, training_labels = zip(*training_data[:10000])
    print np.array(training_features).shape
    training_features = np.squeeze(training_features).transpose()
    training_labels = np.squeeze(training_labels).transpose()
    weights = train(training_labels, training_features,nn,30,30)
    test_features, test_labels = zip(*test_data)
    test_features = np.squeeze(test_features).transpose()
    test_labels = np.squeeze(test_labels).transpose()
    test(test_labels,test_features,weights,nn)


if __name__ == "__main__":
    main()
