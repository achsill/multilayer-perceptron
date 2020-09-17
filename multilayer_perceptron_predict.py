import numpy as np
from multilayer_perceptron_train import Layer, process_file
from utils import *
import argparse

layers = np.load("nn_weights.npy", allow_pickle=True)
parser = argparse.ArgumentParser()
parser.add_argument("file", help="Choose file")
args = parser.parse_args()
if (args.file == None):
    print('Problem with the file')
    exit(1)

X, y = process_file(args.file)
print(cost(layers, X, y))
