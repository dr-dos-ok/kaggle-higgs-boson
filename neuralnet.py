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

	def forward(self, inputs, return_all=False):
		
		#sum of inputs (first layer will be ignored)
		layer_activations = [
			np.empty(size)
			for size in self.layer_sizes
		]

		#after sigmoid function
		layer_outputs = [
			np.empty(size + 1)
			for size in self.layer_sizes
		]

		#set input as output of first layer
		layer_outputs[0][:-1] = inputs
		layer_outputs[0][-1] = 1 # bias neuron

		for prev_layer_index, layer_index in adjacent_pairs(range(self.nlayers)):
			layer_activations[layer_index] = np.dot(
				layer_outputs[prev_layer_index].transpose(),
				self.weights[prev_layer_index]
			)
			layer_outputs[layer_index][:-1] = self.sigmoid(layer_activations[layer_index])
			layer_outputs[layer_index][-1] = 1.0 # bias neuron

		result = layer_outputs[-1][:-1]
		if return_all:
			#useful for debugging and unit tests, mostly
			return (result, layer_outputs, layer_activations)
		else:
			return result