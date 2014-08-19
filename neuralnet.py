import numpy as np
import scipy.special

def adjacent_pairs(a):
	for i in range(len(a)-1):
		yield (a[i], a[i+1])

def logistic(x):
	# return 1.0 / (1.0 + np.exp(-x))
	return scipy.special.expit(x)

def logistic_deriv(x):
	log = logistic(x)
	return log * (1 - log)

NORMAL_OUTPUTS = 0
DEBUGGING_OUTPUTS = 1
ALL_LAYER_OUTPUTS = 2

ONE = np.ones(1)

class FeedForwardNet(object):

	def __init__(
		self,
		layer_sizes,
		alpha=0.1,
		sigmoid_fn_pair=(logistic, logistic_deriv)
	):
		self.alpha = 0.1

		self.sigmoid, self.sigmoid_deriv = sigmoid_fn_pair

		self.layer_sizes = np.array(layer_sizes)
		self.nlayers = len(layer_sizes)

		#+1 for bias
		self.weights = [
			np.random.random((top, bottom+1))
			for bottom, top in adjacent_pairs(layer_sizes)
		]

	def forward(self, inputs, outputs=NORMAL_OUTPUTS):
		
		#sum of inputs (first layer will be ignored)
		layer_activations = [
			np.empty(size)
			for size in self.layer_sizes
		]

		#after sigmoid function
		layer_outputs = [
			np.empty(size)
			for size in self.layer_sizes
		]

		#set input as output of first layer
		layer_outputs[0] = inputs

		for prev_layer_index, layer_index in adjacent_pairs(range(self.nlayers)):
			layer_activations[layer_index] = np.dot(
				np.concatenate([
					layer_outputs[prev_layer_index],
					ONE # bias unit
				]).transpose(),
				self.weights[prev_layer_index]
			)
			layer_outputs[layer_index] = self.sigmoid(layer_activations[layer_index])

		if outputs == NORMAL_OUTPUTS:
			return layer_outputs[-1]
		elif outputs == DEBUGGING_OUTPUTS:
			return (layer_outputs, layer_activations)
		elif outputs == ALL_LAYER_OUTPUTS:
			return layer_outputs
		else:
			raise Exception("Unrecognized value of 'outputs' in FeedForwardNet.forward()")

	def get_partial_derivs(self, test_case):
		"""backprop implementation"""



		# squared error = (1/2) * sum((expected - actual)**2)
		# squared err derive = actual - expected
