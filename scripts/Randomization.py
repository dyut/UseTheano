import numpy
import theano.tensor as T
from theano import function
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams

ranstr = RandomStreams(seed=42)
rv_u = ranstr.uniform((2,2))
rv_n = ranstr.normal((2,2))

rand_uni = function([], rv_u, no_default_updates=True)
rand_norm = function([], rv_n)

print 'Uniform random 2x2 matrix:'
print str(rand_uni())
print 'Uniform random 2x2 matrix:'
print str(rand_uni())
print 'Normal random 2x2 matrix:'
print str(rand_norm())
print 'Normal random 2x2 matrix:'
print str(rand_norm())
