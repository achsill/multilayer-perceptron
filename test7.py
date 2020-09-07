import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from utils import *

np.random.seed(0)
# X, y = datasets.make_moons(200, noise=0.20)


# Import Data
df = pd.read_csv('data.csv')

bar_color = {"Ravenclaw": "#2980b9",
            "Slytherin": "#27ae60",
            "Gryffindor": "#c0392b",
            "Hufflepuff": "#f1c40f"}

# print(len(df.columns))
i = 0
header = []
while i < len(df.columns):
    if (i == 0):
        header.append('Index')
    elif (i == 1):
        header.append('State')
    else:
        header.append('Col-' + str(i))
    i+=1

df.columns = header
index = df['Index']
state = df['State']
df = df.drop(columns=['Index', 'State'])
df = ((df-df.min())/(df.max()-df.min()))
df['Index'] = index
# End import Data

X = np.array(df[['Col-25', 'Col-29', 'Col-9', 'Col-24', 'Col-12', 'Col-4', 'Col-2', 'Col-5']])
# print(X)
# print(df)
# print(state)
y = []
for i in range(0, len(state)):
    if (state[i] == 'M'):
        y.append(1)
    elif (state[i] == 'B'):
        y.append(0)

y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, shuffle=True)

# print('____LA_____')
# print(X_train)
# print('____LA___')
# print(X_test)

# print(y)

# X = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
# y = np.array([1, 1, 0, 0, 0, 0, 1, 1])
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# print(y)

num_examples = len(X_train) # training set size
nn_input_dim = 8 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X_train.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y_train])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    np.set_printoptions(threshold=sys.maxsize)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # print('________')
    # print(probs)
    # print('________')
    return np.argmax(probs, axis=1)


def cross_ent(model, y, x):
    # a = a + 1e-15s
    np.set_printoptions(threshold=sys.maxsize)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    # print("______________")
    a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print(a)
    # print("______________")
    # print(a.T[0])
    # print("______________")
    # print(a.T[1])
    # print("______________")
    if 0 in a.T[1]:
        print('STOP:')
        print(a.T[1][a.count(0)])
        exit(1)
    # return np.sum(np.nan_to_num(-y*np.log(a.T[1])-(1-y)*np.log((1-(a.T[1])))))
    try:
        rez = -(1 / num_examples) * np.sum(np.nan_to_num(np.dot(y, np.log(a.T[1]) - np.dot(1 - y, np.log(1 - (a.T[1]))))))
    except:
        rez = 0
    return rez


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=200000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X_train.dot(W1) + b1
        # print('z1: ', z1)
        a1 = sigmoid(z1)
        # print('a1: ', a1)
        # print('W2: ', W2)
        z2 = a1.dot(W2) + b2
        # print('z2: ', z2)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


        # print('probs: ', probs)
        # Backpropagation
        delta3 = probs
        # print('_______')
        # print(delta3)
        delta3[range(num_examples), y_train] -= 1
        delta3 = delta3 / num_examples
        # print(delta3)
        # print('_______')
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * dSigmoid(a1)
        dW1 = np.dot(X_train.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss: ", i, " - ", calculate_loss(model), '- and - ', cross_ent(model, y_train, X_train))
            # print("Loss: ", i, " - ", calculate_loss(model))
            # print("Loss: ", i, " - ", cross_ent(model, y_train))

    return model


# Build a model with a 3-dimensional hidden layer
model = build_model(12, print_loss=True)
leX = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
# print(lambda x: predict(model, x))

print(predict(model, X_test))
print(y_test)

print(cross_ent(model, y_test, X_test))
# print(cross_ent(predict(model, X_test), y_test))
# Plot the decision boundary
# plot_decision_boundary(predict(model, X_test))
# plt.title("Decision Boundary for hidden layer size 3")
# plt.show()
