import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def dSoftmax(a, y):
    n_samples = y.shape[0]
    res = a - y
    return res/n_samples

def cost(layers, x, y):
    output = x
    output_activated = []
    for layer in layers:
        output_activated, output = layer.forward_propagation(output, output_activated)

    y_predicted = []
    y_true = []
    for i in range(0, y.shape[0]):
            y_true.append(y[i][1])
            y_predicted.append(output_activated[i][1])

    y_predicted = np.array(y_predicted)
    y_true = np.array(y_true)
    rez = -(1 / y.shape[0]) * np.sum(np.nan_to_num(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - (y_predicted))))

    return rez
