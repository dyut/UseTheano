import numpy
import theano
import theano.tensor as T

rng = numpy.random
N=100
feats=3

D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

training_steps=100
regFactor=0.01
lr=0.01

x = T.matrix("x")
y = T.vector("y")

w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0, name="b")

pred_score = 1 / (1 + T.exp(-T.dot(w, x) - b))
pred = pred_score > 0.5

crossEnt = -y * T.log(pred_score) - (1 - y) * T.log(1 - pred_score)
cost = crossEnt.mean() + regFactor * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])

train = theano.function(inputs=[x,y], outputs=[pred, crossEnt], updates=((w, w - lr * gw), (b, b - lr * gb)))
predict = theano.function(inputs=[x], outputs=pred)

for i in range(training_steps):
    pred, err = train(D[0], D[1])


print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

