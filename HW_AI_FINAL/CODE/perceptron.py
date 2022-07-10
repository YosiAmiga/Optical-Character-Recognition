import warnings
import numpy as np
from sklearn.metrics import accuracy_score
import time

warnings.filterwarnings("ignore")


# ----------------------------------
# load data funtions --> different from Naive bayes , here we also flatten the data into 1D.


def _pixel_to_value(character):
    if (character == ' '):
        return 0
    elif (character == '#'):
        return 1
    elif (character == '+'):
        return 2


def _value_to_pixel(value):
    if (value == 0):
        return ' '
    elif (value == 1):
        return '#'
    elif (value == 2):
        return '+'


'''
Function for loading data and label files
'''


def _load_data_file(filename, n, width, height):
    fin = [l[:-1] for l in open(filename).readlines()]
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            row = map(_pixel_to_value, list(fin.pop()))
            data.append(row)
        data = np.array(data)
        data = data.flatten()
        items.append(data)
    return items


# stayed the same
def _load_label_file(filename, n):
    fin = [l[:-1] for l in open(filename).readlines()]
    labels = []
    for i in range(n):
        labels.append(int(fin[i]))
    return labels


# --------------------------------------------------------


DATA_WIDTH = 28
DATA_HEIGHT = 28
NUMBER_OF_TRAINING_EXAMPLES = 5000
NUMBER_OF_VALIDATION_EXAMPLES = 1000
NUMBER_OF_TEST_EXAMPLES = 1000

ALL_TRAINING_IMAGES = []
ALL_TRAINING_LABELS = []

ALL_VALIDATION_IMAGES = []
ALL_VALIDATION_LABELS = []

ALL_TEST_IMAGES = []
ALL_TEST_LABELS = []

# Global Variables

training_size = 5000
testing_size = 1000


"""Actual learning rate"""
alpha = 1.0

"""Since we are not sure that the dataset is linearly separable or not, therefore to avoid infinite looping in weights training
we iterate over the training set epoch times for tuning weights"""

epochs = 5  # epochs for perceptron rule

labels = 10  # numbers in range (0,9)

# -------------------------------------------- LOAD DATA
ALL_TRAINING_IMAGES = _load_data_file("../digitdata/trainingimages",
                                      NUMBER_OF_TRAINING_EXAMPLES, DATA_WIDTH, DATA_HEIGHT)
ALL_TRAINING_LABELS = _load_label_file("../digitdata/traininglabels",
                                       NUMBER_OF_TRAINING_EXAMPLES)

ALL_VALIDATION_IMAGES = _load_data_file("../digitdata/validationimages",
                                        NUMBER_OF_VALIDATION_EXAMPLES, DATA_WIDTH, DATA_HEIGHT)
ALL_VALIDATION_LABELS = _load_label_file("../digitdata/validationlabels",
                                         NUMBER_OF_VALIDATION_EXAMPLES)

ALL_TEST_IMAGES = _load_data_file("../digitdata/testimages",
                                  NUMBER_OF_TEST_EXAMPLES, DATA_WIDTH, DATA_HEIGHT)
ALL_TEST_LABELS = _load_label_file("../digitdata/testlabels",
                                   NUMBER_OF_TEST_EXAMPLES)

# --------------------------------------------------------

"""Predicts the labels by choosing the label of the classifier with highest confidence(probability)"""


def predict(all_weights, test_images, learning_algo):
    test_images = np.hstack((np.ones((testing_size, 1)), test_images))

    predicted_labels = np.dot(all_weights, test_images.T)

    # sigmoid activation function
    if learning_algo == 1:
        predicted_labels = sigmoid(predicted_labels)

    # signum activation function
    elif learning_algo == 2:
        predicted_labels = signum(predicted_labels)

    predicted_labels = np.argmax(predicted_labels, axis=0)

    return predicted_labels.T


# --------------------------------------------------------

"""Activation Functions"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def signum(x):
    x[x > 0] = 1
    x[x <= 0] = -1

    return x


# --------------------------------------------------------
def learn_using_perceptron_rule_with_signum_function(train_images, train_labels, weights):
    epochs_values = []
    error_values = []

    for k in range(epochs):
        missclassified = 0

        for t, l in zip(train_images, train_labels):
            h = np.dot(t, weights)

            h = signum(h)

            if h[0] != l[0]:
                missclassified += 1

            gradient = t * (h - l)  # diff calculated

            # reshape gradient
            gradient = gradient.reshape(gradient.shape[0], 1)

            weights = weights - (gradient * alpha)

        error_values.append(missclassified / training_size)
        epochs_values.append(k)

    global epoch_curve

    return weights

def train(train_images, train_labels, learning_algo):
    # add 1's as x0
    train_images = np.hstack((np.ones((training_size, 1)), train_images))

    # add w0 as 0 initially
    all_weights = np.zeros((labels, train_images.shape[1]))

    train_labels = train_labels.reshape((training_size, 1))

    train_labels_copy = np.copy(train_labels)

    for j in range(labels):

        print("Training Classifier: ", j + 1)

        train_labels = np.copy(train_labels_copy)

        # initialize all weights to zero
        weights = np.zeros((train_images.shape[1], 1))

        if learning_algo == 2:
            for k in range(training_size):
                if train_labels[k, 0] == j:
                    train_labels[k, 0] = 1
                else:
                    train_labels[k, 0] = -1

            weights = learn_using_perceptron_rule_with_signum_function(train_images, train_labels, weights)

        all_weights[j, :] = weights.T

    return all_weights


# --------------------------------------------------------

def run_experiment(train_images, train_labels, test_images, test_labels, learning_algo=2):
    if learning_algo == 2:
        s = "Perceptron Learning Rule"

    print("------------------------------------------------------------------------------------")
    print("Running Experiment using %s" % s)
    print("------------------------------------------------------------------------------------")

    print("Training ...")
    start_time = time.clock()
    all_weights = train(train_images, train_labels, learning_algo)
    print("Training Time: %.2f seconds" % (time.clock() - start_time))
    print("Weights Learned!")

    print("Classifying Test Images ...")
    start_time = time.clock()
    predicted_labels = predict(all_weights, test_images, learning_algo)
    print("Prediction Time: %.2f seconds" % (time.clock() - start_time))

    print("Test Images Classified!")

    accuracy = accuracy_score(test_labels, predicted_labels) * 100  # unnormalize the accuracy

    print("Accuracy: %f" % accuracy, "%")
    print("---------------------\n")


# --------------------------------------------------------

def main():
    """DATA -> pre processing"""
    train_images, train_labels = ALL_TRAINING_IMAGES, ALL_TRAINING_LABELS
    test_images, test_labels = ALL_TEST_IMAGES, ALL_TEST_LABELS
    train_images = np.array(train_images[:training_size])
    train_labels = np.array(train_labels[:training_size], dtype=np.int32)
    test_images = np.array(test_images[:testing_size])
    test_labels = np.array(test_labels[:testing_size], dtype=np.int32)

    """ run one vs all with learning a perceptron with perceptron learning rule """
    run_experiment(train_images, train_labels, test_images, test_labels, 2)


# --------------------------------------------------------

if __name__ == '__main__':

    main()
