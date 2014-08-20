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

def squared_error(actuals, expected):
	diffs = actuals - expected
	return (
		0.5 * np.sum(diffs * diffs), # error
		diffs # derror_by_doutput
	)

NORMAL_OUTPUTS = 0
DEBUGGING_OUTPUTS = 1
ALL_LAYER_OUTPUTS = 2

_ONE = np.ones(1)

class FeedForwardNet(object):

	def __init__(
		self,
		layer_sizes,
		alpha=0.1,
		sigmoid_fn_pair=(logistic, logistic_deriv),
		err_fn=squared_error
	):
		self.alpha = 0.1

		self.sigmoid, self.sigmoid_deriv = sigmoid_fn_pair
		self.err_fn = squared_error

		self.layer_sizes = np.array(layer_sizes)
		self.nlayers = len(layer_sizes)

		self.weights = [
			np.random.random((top, bottom))
			for bottom, top in adjacent_pairs(layer_sizes)
		]
		self.bias_weights = [None] + [
			np.random.random(layer_size)
			for layer_size in self.layer_sizes[1:]
		]

	def forward(self, inputs, outputs=NORMAL_OUTPUTS):
		
		#sum of inputs (first layer will be ignored)
		layer_inputs = [
			np.empty(size)
			for size in self.layer_sizes
		]

		#after sigmoid function
		layer_outputs = [
			np.empty(size)
			for size in self.layer_sizes
		]

		#set input as output of first layer
		layer_outputs[0][:] = inputs

		for prev_layer_index, layer_index in adjacent_pairs(range(self.nlayers)):
			layer_inputs[layer_index] = np.dot(
				layer_outputs[prev_layer_index].T,
				self.weights[prev_layer_index]
			)

			try:
				layer_inputs[layer_index] += self.bias_weights[layer_index]
			except:
				print
				print layer_index
				print self.bias_weights[layer_index]
				print self.bias_weights
				exit()
			layer_outputs[layer_index] = self.sigmoid(layer_inputs[layer_index])

		if outputs == NORMAL_OUTPUTS:
			return layer_outputs[-1]
		elif outputs == DEBUGGING_OUTPUTS:
			return (layer_outputs, layer_inputs)
		elif outputs == ALL_LAYER_OUTPUTS:
			return layer_outputs
		else:
			raise Exception("Unrecognized value of 'outputs' in FeedForwardNet.forward()")

	def get_partial_derivs(self, test_case_inputs, test_case_actuals):
		"""backprop implementation"""

		layer_outputs = self.forward(test_case_inputs, outputs=ALL_LAYER_OUTPUTS)

		layer_output_derivs = [
			np.empty(size)
			for size in self.layer_sizes
		]

		layer_input_derivs = [
			np.empty(size)
			for size in self.layer_sizes[1:]
		]

		weight_derivs = [
			np.empty((top, bottom+1), dtype=np.float64)
			for bottom, top in adjacent_pairs(self.layer_sizes)
		]

		err, derror_by_doutput = self.err_fn(layer_outputs[-1], test_case_actuals)
		layer_output_derivs[-1][:] = derror_by_doutput

		for layer_index in range(self.nlayers-1, 0, -1):
			y = layer_outputs[layer_index]
			layer_input_derivs[layer_index-1] = y * (1.0 - y) * layer_output_derivs[layer_index]

			layer_output_derivs[layer_index-1] = np.dot(
				self.weights[layer_index-1],
				layer_input_derivs[layer_index-1].T
			)

		return (
			[
				weights * layer_input_derivs[index]
				for index, weights in enumerate(self.weights)
			],
			[None] + [
				weights * layer_input_derivs[index]
				for index, weights in enumerate(self.bias_weights[1:])
			]
		)
