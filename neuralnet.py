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

def normalize_vector(vec):
	sq = vec * vec
	return vec / np.sqrt(np.sum(sq))

def flatten_lists_of_arrays(*lists):
	total_size = sum([sum([nda.size for nda in list_]) for list_ in lists])

	result = np.empty(total_size)
	start = 0
	for list_ in lists:
		for ndarray in list_:
			stop = start + ndarray.size
			result[start:stop] = ndarray.ravel()
			start = stop
	return result

def unflatten_to_lists_of_arrays(flattened, *outputs):
	start = 0
	for outlist in outputs:
		for ndarray in outlist:
			stop = start + ndarray.size
			ndarray.ravel()[:] = flattened[start:stop]
			start = stop

NORMAL_OUTPUTS = 0
DEBUGGING_OUTPUTS = 1
ALL_LAYER_OUTPUTS = 2

FLATTENED_OUTPUTS = 3

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
			np.random.random((bottom, top))
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

	def get_flattened_weights(self):
		return flatten_lists_of_arrays(self.weights, self.bias_weights[1:])

	def set_flattened_weights(self, flattened_weights):
		unflatten_to_lists_of_arrays(flattened_weights, self.weights, self.bias_weights[1:])

	def get_partial_derivs(self, test_case_inputs, test_case_actuals, outputs=NORMAL_OUTPUTS):
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

		weight_derivs = [
			weights * layer_input_derivs[index]
			for index, weights in enumerate(self.weights)
		]

		#I usually like to keep a None as the first element so that the indices match
		#up, and to indicate that there are no biases on the first/input layer. However,
		#leaving out the None for a moment makes certain things easier in a sec
		#(flatten_lists_of_arrays doesn't work with None's)
		raw_bias_weight_derivs = [
			weights * layer_input_derivs[index]
			for index, weights in enumerate(self.bias_weights[1:])
		]

		if outputs == NORMAL_OUTPUTS:
			return (weight_derivs, [None] + raw_bias_weight_derivs)
		elif outputs == FLATTENED_OUTPUTS:
			return flatten_lists_of_arrays(weight_derivs, raw_bias_weight_derivs)