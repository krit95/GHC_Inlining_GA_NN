import pickle
import sys
import neat
from neat.six_util import iteritems, iterkeys
from sets import Set
import json

path = "../../"


class Node(object):
  def __init__(self, idx_, activation_, aggregation_, response_, is_input_, bias_, links_=[]):
    self.idx         = idx_
    self.activation  = activation_
    self.aggregation = aggregation_
    self.response    = response_   # Note: When is this used?
    self.is_input    = is_input_
    self.bias        = bias_
    self.links       = links_

  # Expose the dictionaries of the node's links for serialization
  #def __iter__(self):
  #  for link in self.links:
  #    yield link.__dict__

class Link(object):
  def __init__(self, weight_, innode_, outnode_):
    self.weight  = weight_
    self.innode  = innode_
    self.outnode = outnode_


def serialize(obj):
  return obj.__dict__


def toJSON(pkl):
  nodes = {}
  input_nodes = Set([])
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
      if in_node > 0: input_nodes.add(str(in_node))
      out_node = c.key[1]*(-1)
      l = Link(str(c.weight), str(in_node), str(out_node))
      links.append(l)
      #links[in_node].weight   = c.weight
      #links[in_node].out_node = c.out_node 
      #print " in_node: " + str(in_node) + "  out_node: " + str(out_node) + "  weight: " + str(c.weight)
  # print "LINKS: " + str(links)

  # Make all of the input nodes
  for innode in input_nodes:
    nodes[innode] = Node(innode, "identity", "none", "none", True, "0")
 
  def get_links(node):
    nid = node.idx
    if nid in input_nodes:
      return []
    ls = []
    for l in links:
      n = str(l.outnode)
      #print "outnode: " + str(n)
      if n == nid:
        print n + " " + l.innode
        nodes[n].links = get_links(nodes[l.innode])
        l.outnode = n
        ls.append(l)
    print ls
    return ls

  rootnode = nodes["0"]
  rootnode.links = get_links(rootnode)
  rootlink = Link(1, rootnode, None)

  return serialize(rootlink)

with open(path + "pklDumps/genome_102.pkl", "rb") as nnPklRead: 
  g = pickle.load(nnPklRead)
  print toJSON(g)


