import numpy
import theano
import theano.tensor as T
from theano import pp



x = T.scalar("x")
y = x ** 2
gy = T.grad(y,x)
pp(gy)
f = theano.function([x], gy)
print 'f(4) :' + str(f(4))
