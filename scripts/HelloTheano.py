import numpy
import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y

f = function([x,y], z)
sum = f(12, 34.2)
print sum
