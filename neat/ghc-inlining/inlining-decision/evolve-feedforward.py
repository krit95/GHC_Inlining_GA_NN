"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import pickle

import sys

import subprocess

import time
import threading

# Load the config file, which is assumed to live in
# the same directory as this script.
neat_path = "/home/kg95/GHC_Inlining_GA/neat/"
sys.path.append(neat_path)

import neat
import re 

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
current_genome_list = []

no_of_gens = 9

rel_path_proj = "./.."
rel_path_minifib = rel_path_proj + "/minifib"

clean_boot_cmds = ["make clean"]
make_def_ghc_cmd = ["time make MYGHC=ghc MODEL=\"\" 2>&1 | tee baseline"]
baseline_cmds = clean_boot_cmds + make_def_ghc_cmd
make_curr_cmd = " 2>&1 | tee current"
fitness_cmd = "nofib-analyse/nofib-analyse baseline current"

make_w_def_ghc_cmd = "make MYGHC=ghc MODEL=\"\" 2>&1 | tee current" 
make_ghc_command = "time make MYGHC=/home/kg95/ghc-8.6.4/inplace/bin/ghc-stage2 MODEL=\"-model-path "

def run():
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.start_compute(current_genome_list)
    max_fit = 0
    max_fit_id = -1

    os.chdir(rel_path_minifib)
    print("Current dir: ", os.getcwd())

    print("Executing baseline commands")
    for cmd in baseline_cmds:
        print("$$$$$$$$$$$$$$$$$$$ Executing: ", cmd)
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE, shell=True)
        p.communicate()
        # pOut = p.communicate()
        # print("Output: ", pOut[0])
        # print("Error: ", pOut[1])
    # os.chdir("../nofib-analyse")
    # p = subprocess.Popen("", stderr=subprocess.PIPE, shell=True)
    # pOut = p.communicate()
    # print("Output: ", pOut[0])
    # print("Error: ", pOut[1])
    for gen in range(no_of_gens):
        print("Generated genome keys list: " + str(current_genome_list))
	#genome_key = current_genome_list[0]
        for genome_key in current_genome_list:
            print("Current genome: ", genome_key)
            for cmd in clean_boot_cmds:
            	print("$$$$$$$$$$$ Executing: ", cmd)
            	p2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            	p2.communicate()
        # print("Output of make current: ", p2.communicate()[0])
	
	    make_hack_ghc = make_ghc_command + str(genome_key) + "\""  + make_curr_cmd
	    print("$$$$$$$$$$$$$$ Exceuting:", make_hack_ghc)
            t1 = threading.Thread(target=run_compile_cmd, args=(make_hack_ghc,))
	    print("Starting thread")
	    t1.start()
	    fitness_final = 1000
	    startTime = 0
	    while (startTime < 25 * 60):
		print("%%%%%%%%%%%%%%%% Waiting ---------------------------")
		time.sleep(10)
		startTime += 10
		if not t1.isAlive():
		    break
	    if t1.isAlive():
		print("%%%%%%%%%%%% Terminating %%%%%%%%%%%%%%")
		p1_compile.kill()	
		# os.killpg(os.getpgid(p1_compile.pid), signal.SIGTERM)
	    else:
            	# print("Output of make ghc: ", p1.communicate()[0])

	    	print("%%%%%%%%%%%%% Executing: ", fitness_cmd)
            	p3 = subprocess.Popen(fitness_cmd, stdout=subprocess.PIPE, shell=True)
            	fitness, p3Err = p3.communicate()
	    	print("fitness error: ", p3Err)
	    	fitness = fitness.strip().strip(" ")
 
	    	print("Fitness for ", genome_key, ":", fitness)
            
	    	# print("------------DONE-----------")
	    	# print("sys path: ", sys.path)
	   
	    	fit_float = re.search("((\+|-)[\d\.]+)", fitness)
	    	if fit_float is not None:	
		    print("regex search: ", fit_float.groups)
	            if len(fit_float.groups()) > 0:
                    	grp1 = fit_float.group(0)
	    	    	print("Captured group:", grp1)
	    	    	if grp1 is not None:
		            fitness_final = float(grp1)
	    	else:
		    print("Fit float was None!!!")
	    log = "Generation: " + str(gen) + ", Genome key: " + str(genome_key) + ", Fitness: " + str(fitness_final)
	    print("Logging: ", log)
	    f = open("fitness_logs.txt", "a+")
	    f.write(log + "\n")
	    f.close()
	    update_fitness(genome_key, fitness_final)
            # print("Max fitness: ", max_fit, ", id: ", max_fit_id)
        continue_processing()
    get_winner()
	

def run_compile_cmd(cmd):
    global p1_compile
    p1_compile = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    p1Out, p1Err = p1_compile.communicate()
    print("######## Make output: ", p1Out)
    print("######## Make error: ", p1Err)
	
# Called when ghc needs to make an inlining decision
def activate_nn(genome_key, inputs_list):
    with open("pklDumps/genome_" + str(genome_key) + ".pkl", 'rb') as nnPklRead:
        current_genome = pickle.load(nnPklRead)
        current_nn = neat.nn.FeedForwardNetwork.create(current_genome, config)
        return current_nn.activate(inputs_list)

# Called when no-fib has finished executing all programs in suite
# using inlining decision outputs from neural network represented by genome_key
def update_fitness(genome_key, fitness):
    with open(neat_path + "pklDumps/genome_" + str(genome_key) + ".pkl", 'rb') as nnPklRead:
        current_gen = pickle.load(nnPklRead)
        current_gen.fitness -= fitness
    with open(neat_path + "pklDumps/genome_" + str(genome_key) + ".pkl", 'wb') as nnPklDump:
        pickle.dump(current_gen, nnPklDump)

def continue_processing():
    with open(neat_path + "pklDumps/pop.pkl", 'rb') as popPklRead:
        pickle.load(popPklRead).continue_processing(current_genome_list)

def get_winner():
    with open(neat_path + "pklDumps/pop.pkl", 'rb') as popPklRead:
        winner = pickle.load(popPklRead).get_winner()
	print("And the winner is: ", winner.key)

if __name__ == '__main__':
    run()
