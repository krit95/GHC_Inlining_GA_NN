class DuckNetwork(object):
	def __init__(self, inputs, outputs, node_evals):
		self.input_nodes  = inputs       # A list of input IDs
		self.output_nodes = outputs      # A list of output IDs
		self.node_evals   = node_evals   
		self.values       = dict((key, 0.0) for key in inputs + outputs) # Just makes a big dictionary of input and output values

	def activate(self, inputs):
		# ignore the inputs and return randomly 1 or 0.
		return [self.values[i] for i in self.output_nodes]

	@staticmethod
	def create(genome, config):
		""" Receives a genome and pretends it doesn't exist.
		    Does what it wants.
		    It's cool like that. """

		return DuckNetwork([1,2,3],["out"], []) 
