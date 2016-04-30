import numpy
import theano.tensor as T
from theano import function

x = T.dmatrix()
s = 1 / (1 + T.exp(-x))

sigmoid = function([x], s)
result = sigmoid([[2, 5, 4], [5, 4, 6]])
print result
