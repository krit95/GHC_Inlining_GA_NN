import pickle
import sys
import neat
from neat.six_util import iteritems, iterkeys

path = "../../"

def toJSON(pkl):
  nodes = {}
  links = {}

  # Collect links into a dictionary
  connections = list(pkl.connections.values())
  connections.sort()
  for c in connections:
    if c.enabled: # Don't bother with dead connections
      linkname = str(c.key[0]*(-1))
      in_node  = c.key[0]*(-1)
      out_node = c.key[1]*(-1)
      weight   = c.weight
      l = links.setdefault(linkname, {})
      links[linkname]["weight"]  = c.weight
      links[linkname]["outnode"] = str(out_node)
      links[linkname]["innode"]  = str(in_node)
      print linkname + " in_node: " + str(in_node) + "  out_node: " + str(out_node) + "  weight: " + str(c.weight)

  # Collect nodes into a dictionary
  #for k, ng in iteritems(pkl.nodes):
  #  n = nodes.setdefault(str(k*(-1))) # Retrive node from dict; set empty default if not there
  #for k, ng in iteritems(nodes):
  #  s += k
  #return s

with open(path + "pklDumps/genome_102.pkl", "rb") as nnPklRead: 
  g = pickle.load(nnPklRead)
  print toJSON(g)


