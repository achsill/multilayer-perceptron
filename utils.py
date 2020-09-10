import numpy as np
# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

def softmax(X):
    logits_exp = np.exp(X)
    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)
    # exps = np.exp(X - np.max(X))
    # return exps / np.sum(exps)

def dSoftmax(softmax):
    # s = softmax.reshape(-1,1)
    return softmax * (1. - softmax)

def dCrossEntropy(a, y):
     return a - y
