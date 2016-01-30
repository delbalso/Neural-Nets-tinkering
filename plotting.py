
import matplotlib.pyplot as plt


def plot_hyperparam_search(x, y, z, x_label, y_label):
    print x
    print y
    print z
    plt.figure(1)
    plt.title('Hyperparam search')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    #plt.axis([x.min(), 10, y.min(), 10])
    plt.pcolor(x, y, z, cmap='RdBu')
    plt.colorbar()
    plt.show()


def plot_history(
        training_accuracy_history,
        validation_accuracy_history,
        training_cost_history,
        validation_cost_history):
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
