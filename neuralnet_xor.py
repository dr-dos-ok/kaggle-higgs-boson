import numpy as np
import pandas as pd
import neuralnet as nn

from scipy.special import expit

import signal, bkputils, math, sys

BATCH_SIZE = 2

xor = np.array([
	[0, 0, 0],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 0]
])

manual_net = nn.FeedForwardNet([2, 2, 1])
manual_net.weights = [
	np.array([
		[10.0, 10.0],
		[10.0, 10.0]
	]),
	np.array([
		[-10.0],
		[10.0]
	])
]
manual_net.bias_weights = [
	None,
	np.array([-15.0, -5.0]),
	np.array([-5.0])
]
# print manual_net.forward(xor[:,:-1])


###########################################

net = nn.FeedForwardNet([2, 2, 1])
# net.weights = [nda.copy() for nda in manual_net.weights]
# net.bias_weights = [nda.copy() if nda is not None else None for nda in manual_net.bias_weights]
net_weights = net.get_flattened_weights()
velocity = net.zeros_like_flattened_weights()

initial_error = np.mean((xor[:,[2]] - net.forward(xor[:,[0,1]]))**2)
min_err = initial_error
min_weights = net_weights.copy()
print "initial_error:", initial_error

while not bkputils.is_cancelled():
	batch_indices = np.random.choice(xor.shape[0], BATCH_SIZE, replace=False)
	inputs = xor[:,[0,1]][batch_indices]
	output = xor[:,[2]][batch_indices]

	grad = net.get_partial_derivs(
		inputs,
		output,
		outputs=nn.FLATTENED_WEIGHTS
	)

	sqerr = np.mean((xor[:,[2]] - net.forward(xor[:,[0,1]]))**2)

	if sqerr < min_err:
		min_err = sqerr
		min_weights = net_weights.copy()

	learning_rate = 0.02 #sqerr / 1000.0
	velocity_decay = 0.9 # expit(40.0 * sqerr - 4.0)

	grad = grad * learning_rate
	velocity *= velocity_decay
	velocity += -grad

	net_weights += velocity
	any_nan = np.isnan(np.sum(net_weights))
	if not any_nan:
		net.set_flattened_weights(net_weights)

	sys.stdout.write("\rerror: %1.4f     " % sqerr)
	sys.stdout.flush()
print
print "----------"

print "min_err:", min_err

nn.printl("outputs", net.forward(xor[:,[0,1]]))

nn.printl("weights", net.weights)
nn.printl("bias_weights", net.bias_weights)