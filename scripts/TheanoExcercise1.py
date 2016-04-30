import numpy
import theano.tensor as T
from theano import function

x = T.vector()
y = T.vector()
z = x ** 2 + y ** 2 + 2 * x * y

f = function([x,y], z)
wholesq = f([2, 5, 4], [5, 4, 6])
print wholesq
