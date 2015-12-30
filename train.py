# http://neuralnetworksanddeeplearning.com/chap2.html
import math
import pandas as pd
import numpy as np

def logistic (i):
    return 1/(1 + math.exp(-1 * i))
logistic = np.vectorize(logistic, otypes=[np.float])

def accuracy(output, labels):
    output_values = np.argmax(output, axis=0)
    label_values = np.argmax(labels, axis=0)
    return sum(output_values==label_values)/float(len(label_values))


# input layer = a1
# input -> a1 = input * w1 -> a2 = a1 * w2

def train(labels, features):
# initalize
    num_data = labels[1,:].size
    num_labels = labels[:,1].size
    num_features = features[:,1].size
    w1 = np.random.random((num_labels, num_features))/math.sqrt(num_features) - 1/math.sqrt(num_features)/2 # 10 x 257
    for j in range(200):
# train
        output = logistic(np.dot(w1,features)) # 10 x 1593
        training_accuracy = accuracy(output,labels)
        print "training accuracy: " + str(training_accuracy)

# compute error, elementwise computations
        sigma_derivative = np.multiply(logistic(np.dot(w1, features)), 1 - logistic(np.dot(w1, features))) # 10 x 1593
        delta = np.multiply(output - labels, sigma_derivative) # 10 x 1593

     # compute delta W
        weight_deltas = np.zeros((num_data,num_labels,num_features ))
        learning_rate = 1
        for i in range(num_data):
            weight_deltas[i,:,:] = np.dot(np.mat(delta)[:,i],np.mat(features)[:,i].transpose()) # mat multiplication
        delta_w1 = weight_deltas.mean(axis=0)
        w1 = w1 - learning_rate * delta_w1
    return w1

def test(labels, features, weights):
        output = logistic(np.dot(weights,features)) # 10 x 1593
        test_accuracy = accuracy(output,labels)
        print "test accuracy = "+str(test_accuracy)

#read in data
raw_data = pd.read_csv("/Users/delbalso/projects/nn1/data.data", sep=" ", header=None)
print raw_data
raw_data = raw_data.drop(raw_data.columns[[-1]],1)
data = np.ones((raw_data.shape[0],raw_data.shape[1]+1)).transpose()
data[1:,:] = raw_data.transpose()
features = data[:-10,:] # 257 x 1593
labels = data[-10:,:] # 10 * 1593

weights = train(labels[:,:-200],features[:,:-200])
test(labels[:,-200:],features[:,-200:],weights)

