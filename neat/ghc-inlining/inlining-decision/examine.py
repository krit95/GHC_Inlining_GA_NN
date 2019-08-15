import pickle
import sys
import neat
from neat.six_util import iteritems, iterkeys
from sets import Set
import json

path = "../../"


class Node(object):
  def __init__(self, idx_, activation_, aggregation_, response_, is_input_, bias_, links_=[]):
    self.idx         = idx_          # :: String
    self.activation  = activation_   # :: String
    self.aggregation = aggregation_  # :: String
    self.response    = response_     # :: String ... Note: When is this used?
    self.is_input    = is_input_     # :: Bool
    self.bias        = bias_         # :: Double
    self.links       = links_        # :: [Link]

  def __str__(self):
    copy = self
    copy.links = [str(_) for _ in self.links]
    d = str(copy.__dict__)
    return d

class Link(object):
  def __init__(self, weight_, innode_, outnode_):
    self.weight  = weight_     # :: Double
    self.innode  = innode_     # :: Node
    self.outnode = outnode_    # :: String (of an index number)

  def __str__(self):
    copy = self
    copy.innode = str(self.innode)
    d = str(copy.__dict__)
    return d

def toJSON(pkl):
  nodes = {}
  input_nodes = []
  links = []

  # Create hidden nodes
  for k, ng in iteritems(pkl.nodes):
    nodename = str(k*(-1)) # Change node idx's to negative; they're not input nodes
    newnode = Node(nodename, ng.activation, ng.aggregation, ng.response, False, ng.bias)
    nodes[nodename] = newnode

  # Collect links
  connections = list(pkl.connections.values())
  connections.sort()
  for c in connections:
    if c.enabled: # Don't bother with dead connections
      #linkname = str(c.key)
      in_node  = c.key[0]*(-1)
      if in_node > 0: # Input nodes have positive indices
        str_innode = str(in_node)
        if str_innode not in nodes:
          input_nodes.append(str_innode)
          # create the input node & add to nodes
          nodes[str_innode] = Node(str_innode, "identity", "none", "none", True, "0")
      out_node = c.key[1]*(-1)
      l = Link(str(c.weight), nodes[str_innode], str(out_node))
      links.append(l)

  def get_links(node):
    nid = node.idx
    if nid in input_nodes:
      return []
    ls = []
    for l in links:
      n = str(l.outnode)
      #print "outnode: " + str(n)
      if n == nid:
        #print n + " " + l.innode
        l.innode.links = get_links(l.innode)
        ls.append(l)
    #print ls
    return ls

  rootnode = nodes["0"]
  rootnode.links = get_links(rootnode)
  rootlink = Link(1, rootnode, None)

  # Some ugly reformatting seems necessary
  return str(rootnode).replace('"','').replace("\\",'').replace("'{","{").replace("}'","}")

with open(path + "pklDumps/genome_102.pkl", "rb") as nnPklRead: 
  g = pickle.load(nnPklRead)
  print toJSON(g)


