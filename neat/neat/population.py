"""Implements the core evolution algorithm."""
from __future__ import print_function

import pickle

import neat
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

import sys
import os

neat_path = "/home/kg95/GHC_Inlining_GA/neat/"
sys.path.append(neat_path)

class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        print("Path:", sys.path)
	self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
            self.k = 0
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def start_compute(self, current_genome_list):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        if self.config.no_fitness_termination:
            raise RuntimeError("Cannot have no generational limit with no fitness termination")
        self.dump_genomes_pop(current_genome_list)

    def dump_genomes_pop(self, current_genome_list):
	print("Dumping pop and genomes of size: ", len(current_genome_list))
        if self.config.no_fitness_termination:
            print("no fitness termination")
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        for genome in itervalues(self.population):
            genome.fitness = 0

        self.reporters.start_generation(self.generation)

	pop_loc = neat_path + "pklDumps/pop.pkl"

        with open(pop_loc, 'wb') as popPklDump:
            print("Dumping population object to: ", neat_path, "pklDumps/pop")
            pickle.dump(self, popPklDump)

        print("Dumping current genomes")
        del current_genome_list[:]
        for index, genome in iteritems(self.population):
            print("Currently dumping: ", genome.key)
            current_genome_list.append(genome.key)
            with open(neat_path + "pklDumps/genome_" + str(genome.key) + ".pkl", 'wb') as nnPklDump:
                pickle.dump(genome, nnPklDump)
        print("Dump complete")
	if self.best_genome is not None:
            print("best genome now: ", self.best_genome.key)
	else :
	    print("Best genome null")

    # Called when ghc is done with all the neural nets in this generation
    def continue_processing(self, current_genome_list):
        print("Continue processing")
        # self.population = {}
        for genome_key in current_genome_list:
            print("Currently opening: ", genome_key)
	    with open(neat_path + "pklDumps/genome_" + str(genome_key) + ".pkl", 'rb') as nnPklRead:
                current_gen = pickle.load(nnPklRead)
                self.population[genome_key].fitness = current_gen.fitness
        best = None
        for g in itervalues(self.population):
            if best is None or g.fitness > best.fitness:
                best = g
        self.reporters.post_evaluate(self.config, self.population, self.species, best)

	print("Chosen current best as: ", best.key)

        # Track the best genome ever seen.
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best
	
	print("Chosen overall best as: ", self.best_genome.key)

        if not self.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best)
                return

        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species,
                                                      self.config.pop_size, self.generation)

        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)

        self.reporters.end_generation(self.config, self.population, self.species)

        self.generation += 1

        # continue to top of loop for next k
        self.dump_genomes_pop(current_genome_list)

    def get_winner(self):
        print("Get winner", self.best_genome.key)
        return self.best_genome
