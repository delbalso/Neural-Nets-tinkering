# http://neuralnetworksanddeeplearning.com/chap3.html
import pandas as pd
import numpy as np
import random
import mnist_loader as mnist
import matplotlib.pyplot as plt
import copy

TRACK_VALIDATION_ACCURACY = True

# Hyperparameters
MAX_DECLINES_BEFORE_STOP = 10
LEARNING_RATE = .1
L2_LAMDA = 5
FRICTION = 0.9

np.random.seed(12345678)
training_accuracy_history = list()
validation_accuracy_history = list()
training_cost_history = list()
validation_cost_history = list()


def plot_history(training_accuracy_history):
    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }

# accuracy subplot
    plt.figure(1)
    plt.subplot(211)
    plt.plot(training_accuracy_history, 'b-', label="Training Data")
    plt.plot(validation_accuracy_history, 'r-', label="Validation Data")
    plt.title('Model accuracy during training', fontdict=font)
    plt.xlabel('Training Epochs', fontdict=font)
    plt.ylabel('Classification Accuracy (%)', fontdict=font)
    plt.subplots_adjust(hspace=.5)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

# cost subplot
    plt.subplot(212)
    plt.plot(training_cost_history, 'b-', label="Training Data")
    plt.plot(validation_cost_history, 'r-', label="Validation Data")
    plt.title('Cost Function during training', fontdict=font)
    plt.xlabel('Training Epochs', fontdict=font)
    plt.ylabel('Cost', fontdict=font)
    plt.subplots_adjust(hspace=.5)
    plt.ylim([0, plt.ylim()[1]])
    plt.legend(loc='upper right')

# Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


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
    if False:  # Use this to print distributions
        for i in xrange(10):
            output_dist.append((output_values == i).sum())
            label_dist.append((label_values == i).sum())
        print output_dist
        print label_dist
    return sum(output_values == label_values) / float(len(label_values))

# input layer = a0
# input -> a1 = w1 * a0 -> a2 = w2 * a1


def initialize_weights(nn):
    w = dict()
    for i in xrange(1, nn.index_of_final_layer + 1):
        limit = np.sqrt(
            float(6) / (nn.layers[i - 1].size + 1 + nn.layers[i].size))
        w[i] = np.random.uniform(-1 * limit, limit,
                                 (nn.layers[i].size, nn.layers[i - 1].size + 1))
    nn.w = w


class QuadraticCost():

    @staticmethod
    def cost(outputs, labels):
        print outputs.shape[1]
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


def train(
        training_data, validation_data, test_data, nn,
        epochs,
        mini_batch_size,
        costfunction=CrossEntropyCost):
    num_training_data = training_data.labels[1, :].size
    a = dict()
    delta_w = dict()
    delta = dict()
    validation_accuracy = -1
    validation_accuracy_decreases = 0
    initialize_weights(nn)
    velocity = dict()
    for index in nn.w:
        velocity[index] = np.zeros_like(nn.w[index])

    for j in range(epochs):  # number of training loops
        print "Starting epoch " + str(j) + " of " + str(epochs)
        if MAX_DECLINES_BEFORE_STOP > 0:
            old_validation_accuracy = validation_accuracy

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
            if training_accuracy == 1:
                break

        if MAX_DECLINES_BEFORE_STOP > 0 or TRACK_VALIDATION_ACCURACY:
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


        if MAX_DECLINES_BEFORE_STOP > 0:
            if validation_accuracy <= old_validation_accuracy:
                validation_accuracy_decreases += 1
            else:
                validation_accuracy_decreases = 0
            if validation_accuracy_decreases > MAX_DECLINES_BEFORE_STOP:
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

                elif l == nn.index_of_final_layer - 1:
                    delta[l] = np.multiply(
                        np.dot(
                            nn.w[
                                l + 1].transpose(),
                            delta[
                                l + 1]),
                        sigma_derivative)
                else:
                    delta[l] = np.multiply(
                        np.dot(nn.w[l + 1].transpose(), delta[l + 1][1:]), sigma_derivative)

             # compute delta W
                weight_deltas = np.zeros(
                    (this_batch_size,
                     nn.layers[l].size,
                        nn.layers[
                         l - 1].size + 1))
                for datum in range(this_batch_size):
                    if l == nn.index_of_final_layer:
                        weight_deltas[datum, :, :] = np.dot(np.mat(delta[l])[:, datum], np.mat(
                            a[l - 1])[:, datum].transpose())  # mat multiplication
                    else:
                        weight_deltas[datum, :, :] = np.dot(np.mat(delta[l])[1:, datum], np.mat(
                            a[l - 1])[:, datum].transpose())  # mat multiplication
                delta_w[l] = weight_deltas.mean(axis=0)
# Apply regularization
                regularization = 1 - LEARNING_RATE * L2_LAMDA / this_batch_size
                regularization_mat = np.append(
                    [1], np.ones(nn.w[l].shape[1] - 1) * regularization)
                velocity[l] = old_velocity[l] * FRICTION - LEARNING_RATE * delta_w[l]
                nn.w[l] = nn.w[l] * regularization_mat[None, :] + velocity[l]



def feedforward(features, nn):
    a = dict()
    a[0] = np.ones((features.shape[0] + 1, features.shape[1]))
    a[0][1:, :] = features
    for l in nn.w:
        z = np.dot(nn.w[l], a[l - 1])
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


def test(data, nn):
    a = feedforward(data.features, nn)
    output = a[nn.index_of_final_layer]  # 10 x 1593
    # print output
    test_accuracy = accuracy(output, data.labels)
    print "test accuracy = " + str(test_accuracy)


class NN():

    def __init__(self, layers):
        self.layers = layers
        self.index_of_final_layer = len(layers) - 1
        self.w = dict()


class Layer():
    T_LOGISTIC = 'LOGISTIC'
    T_SOFTMAX = 'SOFTMAX'

    def __init__(self, size, ltype=T_LOGISTIC):
        self.size = size
        self.ltype = ltype


class MLDataSet():

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


def main():
    nn = NN([Layer(784), Layer(100), Layer(10, ltype=Layer.T_LOGISTIC)])
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
    training_features, training_labels = zip(*training_data[:1000])
    training_data = MLDataSet(
        np.squeeze(training_features).transpose(),
        np.squeeze(training_labels).transpose())
    validation_features, validation_labels = zip(*validation_data[:1000])
    validation_data = MLDataSet(
        np.squeeze(validation_features).transpose(),
        np.squeeze(validation_labels).transpose())
    test_features, test_labels = zip(*test_data)
    test_data = MLDataSet(
        np.squeeze(test_features).transpose(),
        np.squeeze(test_labels).transpose())

    train(
        training_data,
        validation_data,
        test_data,
        nn,
        30,
        30)

    test(test_data, nn)
    plot_history(training_accuracy_history)


if __name__ == "__main__":
    main()
