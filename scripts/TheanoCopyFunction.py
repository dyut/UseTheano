import numpy
import theano.tensor as T
from theano import function
from theano import shared

s1 = shared(0)
s2 = shared(0)

x = T.iscalar()

accumulator = function([x], s1, updates=[(s1, s1+x)])
result = accumulator(2)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = accumulator(3)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = accumulator(4)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())

new_accu = accumulator.copy(swap={s1:s2})
result = new_accu(2)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = new_accu(3)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
result = new_accu(4)
print 'Result: ' + str(result) + ' s1: ' + str(s1.get_value()) + ' s2: ' + str(s2.get_value())
