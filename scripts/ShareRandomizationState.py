import numpy
import theano.tensor as T
from theano import function
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

class Graph():
    def __init__(self, seed=123):
        self.rng = RandomStreams(seed)
        self.y = self.rng.uniform((1,))

def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RandomStreams):
        g2.rng.rstate = g2.rng.rstate 
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())


g1 = Graph(seed = 42)
f1 = function([], g1.y)

g2 = Graph(seed = 67)
f2 = function([], g2.y)

print 'f1(): ' + str(f1())
print 'f2(): ' + str(f2())

print 'f1(): ' + str(f1())
print 'f2(): ' + str(f2())

print 'Copy g1 to g2'
copy_random_state(g1, g2)

print 'f1(): ' + str(f1())
print 'f2(): ' + str(f2())

print 'f1(): ' + str(f1())
print 'f2(): ' + str(f2())
