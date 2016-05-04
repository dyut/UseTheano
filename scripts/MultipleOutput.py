import numpy
import theano.tensor as T
from theano import function

x, y = T.dmatrices('x', 'y')
diff = x - y
abs_diff = abs(diff)
diff_sq = diff ** 2
all_diff = function([x, y], [diff, abs_diff, diff_sq])
[d, ad, d_s] = all_diff([[2, 1, 4], [0, 4, -1]], [[10, 0, 3], [-4, 2, -1]])
print 'Diff:'
print d
print 'Abs diff:'
print ad
print 'Diff squared:'
print d_s