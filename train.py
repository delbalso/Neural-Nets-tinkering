# http://neuralnetworksanddeeplearning.com/chap2.html
import pandas as pd
import numpy as np

def logistic (i):
    return 1/(1 + np.exp(-1 * i))
logistic = np.vectorize(logistic, otypes=[np.float])

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
    return sum(output_values==label_values)/float(len(label_values))

# input layer = a1
# input -> a1 = input * w1 -> a2 = a1 * w2

def train(labels, features, nn):
# initalize
    num_data = labels[1,:].size
    a = dict()
    a[0] = features
# set up inital weights
    w = dict()
    delta_w = dict()
    delta = dict()
    for i in xrange(1,nn.index_of_final_layer+1):
        limit = np.sqrt(float(6)/(nn.layers[i-1]+1+nn.layers[i]))
        w[i] = np.random.uniform(-1 * limit, limit, (nn.layers[i], nn.layers[i-1]+1))

    for j in range(500): # number of training loops
        a = feedforward (features, w, nn)
        if j%10==0:
            training_accuracy = accuracy(a[nn.index_of_final_layer],labels)
            print "training accuracy: " + str(training_accuracy)
        for l in sorted(w.keys(),reverse=True):
# compute error, elementwise computations
            sigma_derivative = np.multiply(a[l], 1 - a[l]) # 10 x 1593

            if l == nn.index_of_final_layer:
                delta[l] = np.multiply(a[l] - labels, sigma_derivative) # 10 x 1593
            elif l== nn.index_of_final_layer-1:
                delta[l] = np.multiply(np.dot(w[l+1].transpose(),delta[l+1]), sigma_derivative) # 10 x 1593
            else:
                delta[l] = np.multiply(np.dot(w[l+1].transpose(),delta[l+1][1:]), sigma_derivative) # 10 x 1593

         # compute delta W
            weight_deltas = np.zeros((num_data,nn.layers[l],nn.layers[l-1]+1))
            learning_rate = 1
            for datum in range(num_data):
                if l == nn.index_of_final_layer: # this part could be fucked
                    weight_deltas[datum,:,:] = np.dot(np.mat(delta[l])[:,datum],np.mat(a[l-1])[:,datum].transpose()) # mat multiplication
                else:
                    weight_deltas[datum,:,:] = np.dot(np.mat(delta[l])[1:,datum],np.mat(a[l-1])[:,datum].transpose()) # mat multiplication
            delta_w[l] = weight_deltas.mean(axis=0)
            w[l] = w[l] - learning_rate * delta_w[l]
    return w

def feedforward(features, w, nn):
#data = np.ones((raw_data.shape[0],raw_data.shape[1]+1)).transpose()
#data[1:,:] = raw_data.transpose()
    a = dict()
    a[0] = np.ones((features.shape[0]+1,features.shape[1]))
    a[0][1:,:] = features
    for l in w:
        #print "shape of w["+str(l)+"]: " +str(w[l].shape)
        #print "shape of a["+str(l-1)+"]: " +str(a[l-1].shape)
        no_bias_a = logistic(np.dot(w[l],a[l-1]))
        if l == nn.index_of_final_layer:
            a[l] = no_bias_a
        else:
            a[l] = np.ones((no_bias_a.shape[0]+1,no_bias_a.shape[1]))
            a[l][1:,:] = no_bias_a
        #print (a)
        #print w
    return a

def test(labels, features, weights, nn):
        a = feedforward (features, weights, nn)
        output = a[nn.index_of_final_layer] # 10 x 1593
        #print output
        test_accuracy = accuracy(output,labels)
        print "test accuracy = "+str(test_accuracy)

class NN():
    def __init__(self, layers):
        self.layers = layers
        self.index_of_final_layer = len(layers)-1

def main():
    nn = NN([256, 100, 10])
    print nn.index_of_final_layer
#read in data
    raw_data = pd.read_csv("/Users/delbalso/projects/nn1/data.data", sep=" ", header=None)
    raw_data = raw_data.reindex(np.random.permutation(raw_data.index))
    data = np.array(raw_data.drop(raw_data.columns[[-1]],1).transpose())
    features = data[:-10,:] # 257 x 1593
    labels = data[-10:,:] # 10 * 1593
    weights = train(labels[:,:-200],features[:,:-200],nn)
    test(labels[:,-200:],features[:,-200:],weights,nn)
if __name__ == "__main__":
    main()
