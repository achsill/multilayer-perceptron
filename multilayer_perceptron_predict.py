import numpy as np
from multilayer_perceptron_train import Layer, process_file
from utils import *
import argparse
import os.path
from os import path

if __name__ == "__main__":
    if (path.exists("nn_weights.npy") == False):
        print('Please train the model before trying to predict')
        exit(1)
    try:
        layers = np.load("nn_weights.npy", allow_pickle=True)
    except:
        print('Error with the weights file')
        exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Choose file")
    args = parser.parse_args()
    if (args.file == None):
        print('Problem with the file')
        exit(1)
    elif (path.exists(args.file) == False):
        print('Please choose an existing file')
        exit(1)
    X, y = process_file(args.file)
    print(cost(layers, X, y))
