import csv
import os
import sys
import pickle
import neat

csv_path = "notes/countinlines.csv"

neat_path = "neat/"
sys.path.append(neat_path)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

def run_test():
    with open(csv_path, 'r') as csvFile:
        data = list(csv.reader(csvFile))
        with open(neat_path + "pklDumps/genome_" + str(101) + ".pkl", 'rb') as nnPklRead:
            current_genome = pickle.load(nnPklRead)
            # print("Genome loaded")
            current_nn = neat.nn.FeedForwardNetwork.create(current_genome, config)
            # print("Network created")
        for row in data:
            print("Input: ", row)
            input_list_float = [float(x) for x in row]
            print("Activation output:", current_nn.activate(input_list_float)[0])


run_test()