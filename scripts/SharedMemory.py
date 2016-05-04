import numpy
import theano.tensor as T
from theano import function
from theano import shared

s1 = shared(0)
s2 = shared(0)

x = T.iscalar()
y = T.iscalar()

z = s1 + s2 - x - y

f = function([x, y], z, updates=[(s1, s1+x), (s2, s2 + y)])
result = f(2,3)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = f(3,4)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = f(4,5)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())

s1.set_value(-9)
s2.set_value(-12)
result = f(2,3)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = f(3,4)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())

w = s1 * s2 + x
f2 = function([x, y], w, givens=[(s1, 3), (s2, y)])
result = f2(5,5)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = f2(4,2)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
