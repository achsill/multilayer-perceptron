import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from utils import *

np.random.seed(0)

# Import Data
df = pd.read_csv('data.csv')

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
test = pd.get_dummies(state)
print('t dummy la ?:', test)
df = df.drop(columns=['Index', 'State'])
df = ((df-df.min())/(df.max()-df.min()))
df['Index'] = index
# End import Data

X = np.array(df[['Col-25', 'Col-29', 'Col-9', 'Col-24', 'Col-12']])
# print(X)
# print(df)
# print(state)
# y = []
# for i in range(0, len(state)):
#     if (state[i] == 'M'):
#         y.append(1)
#     elif (state[i] == 'B'):
#         y.append(0)
#
# y = np.array(y)
y = np.array(pd.get_dummies(state))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=True)

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
nn_input_dim = 5 # input layer dimensionality
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
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)

    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    z3 = a2.dot(W3) + b3
    probs = softmax(z3)
    # print('________')
    # print(probs)
    # print('________')
    return np.argmax(probs, axis=1)


def cross_ent(model, y, x):
    # a = a + 1e-15s
    np.set_printoptions(threshold=sys.maxsize)
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)

    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    z3 = a2.dot(W3) + b3
    a = softmax(z3)
    # print("______________")
    # a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # print("______________")
    # print(a.T[0])
    # print("______________")
    # print(a.T[1])
    # print("______________")

    # return np.sum(np.nan_to_num(-y*np.log(a.T[1])-(1-y)*np.log((1-(a.T[1])))))
    # print('LA GROS:')
    # print(y)
    # print('NN LA GROS:')
    # print(a)
    # print('_____s')
    # print(y * np.log(a))
    # print('___________')
    # print((1 - y) * np.log(1 - (a)))
    # print('___________')
    # print(y * np.log(a) + (1 - y) * np.log(1 - a))
    # print('__result___')
    rez = -(1 / len(x)) * np.sum(y * np.log(a + 1e-15) + (1 - y) * np.log(1 - (a + 1e-15)))

    # rez = -(1 / num_examples) * np.sum(np.dot(y, np.log(a.T[1])) + np.dot(1 - y, np.log(1 - (a.T[1]))))

    return np.array(rez)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations

def fw(x, w, b, activation_function):
    z = np.dot(x, w) + b
    a = activation_function(z)
    return a

def build_model(nn_hdim, num_passes=40000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    # W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    # b2 = np.zeros((1, nn_hdim))
    W3 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b3 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X_train.dot(W1) + b1
        a1 = sigmoid(z1)
        # print(a1, '___________s1')

        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)
        # print(a2, '___________s2')

        z3 = a2.dot(W3) + b3
        delta3 = softmax(z3)
        # print(delta3, '___________s3')


        # print('probs: ', probs)
        # Backpropagation
        # print('_______')
        # print(delta3)
        # delta3[range(num_examples), y_train] -= 1
        # delta3 = delta3 / num_examples
        # delta3 = dCrossEntropy(delta3, y_train)
        # print(delta3)
        # print('_______')
        # dW3 = np.dot(a2.T, delta3)
        # db3 = np.sum(delta3, axis=0, keepdims=True)



# BackPropagation on reessaie

        # print('LA:', delta3)
        activation_o = dSoftmax(delta3, y_train)
        i_err3 = np.dot(activation_o, W3.T)
        w_err3 = np.dot(a2.T, activation_o)
        W3 -= epsilon * w_err3
        b3 -= epsilon * np.sum(activation_o, axis=0)

        activation_h2 = dSigmoid(z2) * i_err3
        i_err2 = np.dot(activation_h2, W2.T)
        w_err2 = np.dot(a1.T, activation_h2)
        W2 -= epsilon * w_err2
        b2 -= epsilon * np.sum(activation_h2, axis=0)

        activation_h1 = dSigmoid(z1) * i_err2
        i_err1 = np.dot(activation_h1, W1.T)
        w_err1 = np.dot(X_train.T, activation_h1)
        W1 -= epsilon * w_err1
        b1 -= epsilon * np.sum(activation_h1, axis=0)




# BackPropagation on reessaie




        # o3_error = dCrossEntropy(delta3, y_train)
        # i3_error = np.dot(o3_error, W3.T)
        # w3_error = np.dot(a2.T, o3_error)
        # W3 -= epsilon * w3_error
        # b3 -= epsilon * np.sum(o3_error, axis=0, keepdims=True)
        #
        # o2_error = dSigmoid(z2)
        # i2_error = np.dot(o2_error, W2.T)
        # w2_error = np.dot(a1.T, o2_error)
        # W2 -= epsilon * w2_error
        # b2 -= epsilon * np.sum(o2_error, axis=0)
        #
        # o1_error = dSigmoid(z1)
        # i1_error = np.dot(o1_error, W1.T)
        # w1_error = np.dot(X_train.T, o1_error)
        # W1 -= epsilon * w1_error
        # b1 -= epsilon * np.sum(o1_error, axis=0)

        # delta1 = np.dot(delta3, W2.T) * dSigmoid(z2)
        # dW1 = np.dot(a1.T, delta1)
        # db1 = np.sum(delta1, axis=0)



        # delta2 = np.dot(delta3, W3.T) * dSigmoid(z2)
        # dW2 = np.dot(a1.T, delta2)
        # db2 = np.sum(delta2, axis=0, keepdims=True)
        #
        # delta1 = np.dot(delta2, W2.T) * dSigmoid(z1)
        # dW1 = np.dot(X_train.T, delta1)
        # db1 = np.sum(delta1, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        # dW2 += reg_lambda * W2
        # dW1 += reg_lambda * W1

        # Gradient descent parameter update
        # W1 -= epsilon * dW1
        # b1 -= epsilon * db1
        # W2 -= epsilon * dW2
        # b2 -= epsilon * db2
        # W3 -= epsilon * dW3
        # b3 -= epsilon * db3

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss: ", i , '-', cross_ent(model, y_train, X_train), 'predicted: ', cross_ent(model, y_test, X_test))
            # print("Loss: ", i, " - ", calculate_loss(model))
            # print("Loss: ", i, " - ", cross_ent(model, y_train))

    return model


# Build a model with a 3-dimensional hidden layer
model = build_model(9, print_loss=True)
leX = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
# print(lambda x: predict(model, x))

print(predict(model, X_test))
print('__________')
print(np.argmax(y_test, axis=1))

print('cost = ', cross_ent(model, y_test, X_test))
# Plot the decision boundary
# plot_decision_boundary(predict(model, X_test))
# plt.title("Decision Boundary for hidden layer size 3")
# plt.show()
