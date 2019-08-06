import os
import sys
import pickle
import neat

input_list_str = sys.argv[2]
genome_key_arg = sys.argv[1]

neat_path = "/home/kg95/GHC_Inlining_GA/neat/"
sys.path.append(neat_path)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

def activate_nn(input_list, genome_key):
    #print("Activating: ", genome_key, "on", input_list)
    with open(neat_path + "pklDumps/genome_" + str(genome_key) + ".pkl", 'rb') as nnPklRead:
        current_genome = pickle.load(nnPklRead)
	#print("Genome loaded")
        current_nn = neat.nn.FeedForwardNetwork.create(current_genome, config)
	#print("Network created")
        print(current_nn.activate(input_list)[0])

input_list_float = [float(x) for x in input_list_str.split(",")]

activate_nn(input_list_float, genome_key_arg)
