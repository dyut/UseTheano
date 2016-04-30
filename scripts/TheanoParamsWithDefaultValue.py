import numpy
import theano.tensor as T
from theano import function
from theano import In

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + 2 * y) * w
f = function([In(x, value = 0), In(y, value = 0, name='y_name'), In(w, value = 1)], z)
print 'f(): ' + str(f())
print 'f(4): ' + str(f(4))
print 'f(y_name=3): ' + str(f(y_name=3))
print 'f(4, 3, 2): ' + str(f(4, 3, 2))
print 'f(4, 3): ' + str(f(4, 3))
print 'f(y_name=3, w = 4): ' + str(f(y_name=3, w = 4))