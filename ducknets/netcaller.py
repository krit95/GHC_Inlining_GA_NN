# Loads a saved Neat NN from file and calls its activation function with given inputs
# Ducked example: python netcaller.py ducknet "1,2,3,4" -> 0.0

import pickle
import sys

netname = sys.argv[1] + ".pkl"  # Name of NN. Append .pkl
inputstr = sys.argv[2]            # String of inputs, e.g.: "in1,in2,in3"
ducknet = {}

# Load the neural network from pickle file
with open(netname, 'rb') as input:
	ducknet = pickle.load(input)

inputs = inputstr.split(',')

print ducknet.activate(inputs)[0]
