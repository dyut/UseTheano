import numpy
import theano.tensor as T
from theano import function

x = T.dmatrix()
s = 1 / (1 + T.exp(-x))
g = s * (1 - s)

sigmoid = function([x], [s, g])
[value, gradient] = sigmoid([[2, 1, 4], [0, 4, -1]])
print 'Value:'
print value
print 'Gradient:'
print gradient
