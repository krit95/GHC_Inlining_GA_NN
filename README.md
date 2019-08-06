# Optimizing Compiler Inlining with Machine Learning

## Overview
We used [Neat Python](https://github.com/CodeReclaimers/neat-python), an open-sourced repository for the [NEAT](https://neat-python.readthedocs.io/en/latest/index.html) algorithm implemented in Python.

Each neural network created took in 43 features. The networks were allowed up to one hidden layer and output one float between zero and one. There were 9 epochs total, each creating 20 neural networks. After creation, the networks were pickled and written to file. The Python controller then called the custom GHC and passed each network ID to it through a custom-added compiler flag called -model-path. GHC then re-routed decision making for its non-trivial inlinings through the indicated neural network for the entirety of each compilation, producing an executable for each model. Compilation taking longer than 25 minutes was terminated, and the responsible networks were assigned a fitness value of -1000. This decision was made for the sake of this project’s deadline; future models will receive more examination to vary the penality fitness value—as some networks are far less performant than others, and evolution will be better informed with more specific evaluations. This is especially true considering that early generations produced far more maximum-penalty networks than later generations did.

The Python controller then called our benchmark framework (modified NoFib) to calculate the fitness function as a measure of speedup compared to regular GHC. Per the NEAT algorithm, groups of topologically similar models were grouped together—in a process known as speciation—to preserve innovative network configurations. The models were then bred and mutated for the next generation, where higher-performing models were allowed to produce more offsprings. 

We report our winner as the network which produced the greatest speedup; however, we note that a more in-depth examination of fitness will be needed in future improvements.

The GHC was modified for this use by [Celeste Hollenbeck](https://github.com/CAHollenbeck) and can be found [here](https://github.com/CAHollenbeck/ghc-inlining-study/tree/project_debug_branch).

## Getting started
### Requirements
- [ ] Python 2.7
- [ ] [Modified GHC](https://github.com/CAHollenbeck/ghc-inlining-study/tree/project_debug_branch)

### Running the code
First, clone or download this repo.

#### Simple test
For quick testing, the script "simple_test.py" located [here](https://github.com/krit95/GHC_Inlining_GA_NN/tree/master/neat/ghc-inlining/inlining-decision) prints out the output by reading input from [countinlines.csv](https://github.com/krit95/GHC_Inlining_GA_NN/tree/master/notes) and activating the winner network(id: 101) from our tests.

*You don't need the modified GHC for this, as the inputs have already been extracted by us from an actual run into the specified file*

#### Full test
The primary controller of the whole algorithm is in "evolve-feedforward.py" located [here](https://github.com/krit95/GHC_Inlining_GA_NN/tree/master/neat/ghc-inlining/inlining-decision). 

To run from command line, cd to this repo's root folder and run:
> python ./neat/ghc-inlining/inlining-decision/evolve-feedforward.py

*Make sure you have added python 2.7 to the system path*
