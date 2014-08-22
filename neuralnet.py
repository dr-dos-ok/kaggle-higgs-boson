import numpy as np
import scipy.special
import math

def adjacent_pairs(a):
	"""
	Given a list 'a', return all adjacent pairs of items
	in the list, starting at (a[0], a[1]) and continuing
	on. 'a' must contain at least 2 items.

	This method is a generator method.

	Example: adjacent_pairs([1,2,3]) yields two tuples: (1, 2) and (2, 3)
	"""
	for i in range(len(a)-1):
		yield (a[i], a[i+1])

def logistic(x):
	"""
	Calculate the logistic function for each element in a
	numpy.ndarray.
	http://en.wikipedia.org/wiki/Logistic_function

	Note: this eventually became a simple wrapper for
	scipy.special.expit which does the same thing
	http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.expit.html
	"""
	return scipy.special.expit(x)

def logistic_deriv(x):
	"""
	Calculate the derivative of the logistic function at each
	value of a numpy.ndarray

	If the logistic function is y = 1.0 / (1.0 + exp(x))
	then the derivative of that function can be written as:
	y' = y * (1 - y)
	"""
	log = logistic(x)
	return log * (1 - log)

def squared_error(actuals, expected):
	"""
	Given a 1D numpy.ndarray of calculated values and a same-sized
	numpy.ndarray of expected values, calculate the squared error
	between the two arrays, and the partial derivatives of the error
	with respect to each element of the output.

	Alternately, given 2 same-sized 2D ndarrays of actual and
	expected values, compute a 1D array of errors such that
	errors[i] = squared_error(actuals[i], expected[i])
	The partial derivatives then become a 2D array of the same
	size as the input

	The value returned from this function is a tuple of
	(sq_err, derr_by_doutput)
	with types (float, numpy.ndarray) respectively

	The squared error is given as:
	sq_err = 0.5 * sum((actuals[i] - expected[i])**2)

	The partial derivatives are given as:
	derr_by_doutput[i] = actuals[i] - expected[i]
	"""

	diffs = actuals - expected
	return (
		0.5 * np.sum(diffs * diffs, axis=1), # error
		diffs # derror_by_doutput
	)

def quadratic_error(actuals, expected):

	diffs = actuals - expected
	return (
		np.sum(diffs**4, axis=1),
		-4.0 * (diffs**3)
	)

def normalize_vector(vec):
	"""
	Normalize any vector to a unit vector by dividing by its
	length (Euclidean distance).
	"""
	sq = vec * vec
	return vec / np.sqrt(np.sum(sq))

def sum_sizes(lists):
	"""
	lists is assumed to be a list of lists of numpy.ndarrays (like you
	would get from the *lists parameter in flatten_lists_of_arrays).
	Return the sum of the sizes of all of the ndarrays passed.
	"""
	return sum([sum([nda.size for nda in list_]) for list_ in lists])

def flatten_lists_of_arrays(*lists):
	"""
	Given an arbitrary number of parameters that are all
	lists of numpy.ndarray, return a 1D numpy.ndarray
	that contains the concatenation of all of the input
	ndarrays, flattened
	"""

	total_size = sum_sizes(lists)

	result = np.empty(total_size)
	start = 0
	for list_ in lists:
		for ndarray in list_:
			stop = start + ndarray.size
			result[start:stop] = ndarray.ravel()
			start = stop
	return result

def unflatten_to_lists_of_arrays(flattened, *outputs):
	"""
	This function is intended to be the inverse of flatten_lists_of_arrays(*lists)

	Given a 1D numpy.ndarray and an arbitrary number of parameters that are
	lists of numpy.ndarray, fill each ndarray in order from the contents of the
	1D array.

	This function requires you to initialize and pass the arrays to be filled
	so as to avoid requiring you to pass a complicated object specifying the
	sizes of the nested lists and ndarrays. It is assumed that
	flattened.size == sum_sizes(outputs), but no checking is done to
	verify that.
	"""
	start = 0
	for outlist in outputs:
		for ndarray in outlist:
			stop = start + ndarray.size
			ndarray.ravel()[:] = flattened[start:stop]
			start = stop

def printl(name, larray):
	print name
	for arr in larray:
		print arr
	print

NORMAL_OUTPUTS = 0
DEBUGGING_OUTPUTS = 1
ALL_LAYER_OUTPUTS = 2

FLATTENED_OUTPUTS = 3

_ONE = np.ones(1)

class FeedForwardNet(object):

	def __init__(
		self,
		layer_sizes,
		sigmoid_fn_pair=(logistic, logistic_deriv),
		err_fn=squared_error
	):
		"""
		layer_sizes: (required) a list of ints specifying the number of
		neurons in each layer of this network. There must be at least
		2 ints: the input and output layers respectively. Any additional
		layers will be hidden layers.

		sigmoid_fn_pair: (optional) a 2-tuple of functions representing
		a sigmoid function and its derivative respectively. They are
		both assumed to take in a numpy.ndarray of floats and return
		the same. Default is (neuralnet.logistic, neuralnet.logistic_deriv)

		err_fn: (optional) a function that calculates the error between
		the output of this network and the expected output of a training
		case. The function is assumed to take in two equally-sized
		numpy.ndarrays: (actuals, expected). It is assumed to return a
		2-tuple of (error, derror_by_doutput) with types (float, ndarray)
		respectively. The default value is neuralnet.squared_error
		"""

		self.sigmoid, self.sigmoid_deriv = sigmoid_fn_pair
		self.err_fn = squared_error

		self.layer_sizes = np.array(layer_sizes)
		self.nlayers = len(layer_sizes)

		self.weights = [
			(2.0 * (np.random.random((bottom, top)) - 0.5)) / math.sqrt(bottom+1)
			for bottom, top in adjacent_pairs(layer_sizes)
		]
		self.bias_weights = [None] + [
			(2.0 * (np.random.random(top) - 0.5)) / math.sqrt(bottom+1)
			for bottom, top in adjacent_pairs(layer_sizes)
		]

	def forward(self, inputs, outputs=NORMAL_OUTPUTS):
		"""
		Run the inputs through this FeedForward network in the standard
		normal direction.

		Setting outputs=NORMAL_OUTPUTS (default) will return the outputs
		of the final layer of the network.

		Setting outputs=DEBUGGING_OUTPUTS will return a tuple of
		(layer_outputs, layer_inputs) where layer outputs is a list
		of numpy.ndarray representing the outputs of every neuron at
		every layer of the network. layer_inputs is similar, and of the
		same shape, but contains the inputs to each neuron in the net

		Setting outputs=ALL_LAYER_OUTPUTS will return layer_outputs as
		described in the section for outputs=DEBUGGING_OUTPUTS
		"""
		
		#sum of inputs (first layer will be ignored)
		layer_inputs = [None] * len(self.layer_sizes)

		#after sigmoid function
		layer_outputs = [None] * len(self.layer_sizes)

		#set input as output of first layer
		layer_outputs[0] = inputs.copy()

		for prev_layer_index, layer_index in adjacent_pairs(range(self.nlayers)):
			layer_inputs[layer_index] = np.dot(
				layer_outputs[prev_layer_index],
				self.weights[prev_layer_index]
			)

			layer_inputs[layer_index] += self.bias_weights[layer_index]
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
		"""
		Return a 1D numpy.ndarray that represents the flattened weights of this
		network.

		The structure of the flattened array is not specified to outside code, except
		to guarantee that it is consistent with other flattened_weights methods, and
		that you can apply element-wise operations in a meaningful way (e.g. you can
		add two flattened weight arrays to get the sum of the weights)
		"""
		return flatten_lists_of_arrays(self.weights, self.bias_weights[1:])

	def set_flattened_weights(self, flattened_weights):
		"""
		Set the weights of this network according to the passed flattened_weights

		The structure of the flattened array is not specified to outside code, except
		to guarantee that it is consistent with other flattened_weights methods, and
		that you can apply element-wise operations in a meaningful way (e.g. you can
		add two flattened weight arrays to get the sum of the weights)
		"""
		unflatten_to_lists_of_arrays(flattened_weights, self.weights, self.bias_weights[1:])

	def zeros_like_flattened_weights(self):
		"""
		Return a 1D numpy.zeros array that has the same shape as the arrays used
		in other flattened_weights methods.

		The structure of the flattened array is not specified to outside code, except
		to guarantee that it is consistent with other flattened_weights methods, and
		that you can apply element-wise operations in a meaningful way (e.g. you can
		add two flattened weight arrays to get the sum of the weights)
		"""
		return np.zeros(sum_sizes([self.weights, self.bias_weights[1:]]))

	def get_partial_derivs(self, test_case_inputs, test_case_actuals, outputs=NORMAL_OUTPUTS):
		"""
		Given a single test case and expected outputs for that test case, calculate the
		partial derivatives of the error with respect to the weights of the network.

		Alternately, given a matrix of test cases (test cases are rows), calculate the average
		partial derivatives of each weight over the entire passed set of cases

		Internally this method implements the backpropagation algorithm to calculate the
		weights.
		http://en.wikipedia.org/wiki/Backpropagation
		https://class.coursera.org/neuralnets-2012-001/lecture/39

		If outputs=NORMAL_OUTPUTS (default) is specified, a tuple of
		(weight_derivs, bias_weight_derivs) is returned where each item is a list of
		numpy.ndarray. The shape and order of weight_derivs will be identical to that
		of self.weights and similarly bias_weight_derivs will match up to self.bias_weights

		If outputs=FLATTENED_OUTPUTS is specified, the outputs will be a 1D
		numpy.ndarray that is compatible with the 1D arrays used by the other
		flattened_weights methods available in this class
		"""

		if test_case_inputs.ndim == 1:
			test_case_inputs = test_case_inputs.reshape(1,-1)

		layer_outputs = self.forward(test_case_inputs, outputs=ALL_LAYER_OUTPUTS)

		#init python arrays to appropriate length
		#we'll fill with ndarrays presently
		layer_output_derivs = [None] * len(self.layer_sizes)
		layer_input_derivs = [None] * len(self.layer_sizes)
		weight_derivs = [None] * len(self.layer_sizes - 1)

		err, derror_by_doutput = self.err_fn(layer_outputs[-1], test_case_actuals)
		layer_output_derivs[-1] = derror_by_doutput

		for layer_index in range(self.nlayers-1, 0, -1):
			y = layer_outputs[layer_index]
			layer_input_derivs[layer_index] = y * (1.0 - y) * layer_output_derivs[layer_index]

			layer_output_derivs[layer_index-1] = np.dot(
				layer_input_derivs[layer_index],
				self.weights[layer_index-1].T
			)

		weight_derivs = [
			np.dot(layer_outputs[index].T, layer_input_derivs[index+1]) / layer_outputs[index].shape[0]
			# layer_outputs[index] * np.mean(layer_input_derivs[index+1], axis=0)
			for index, weights in enumerate(self.weights)
		]

		#I usually like to keep a None as the first element so that the indices match
		#up, and to indicate that there are no biases on the first/input layer. However,
		#leaving out the None for a moment makes certain things easier in a sec
		#(flatten_lists_of_arrays doesn't work with None's)
		raw_bias_weight_derivs = [
			np.mean(layer_input_derivs[index+1], axis=0)
			for index, weights in enumerate(self.bias_weights[1:])
		]

		if outputs == NORMAL_OUTPUTS:
			return (weight_derivs, [None] + raw_bias_weight_derivs)
		elif outputs == FLATTENED_OUTPUTS:
			return flatten_lists_of_arrays(weight_derivs, raw_bias_weight_derivs)
		elif outputs == DEBUGGING_OUTPUTS:
			return (layer_input_derivs, layer_output_derivs, weight_derivs, [None] + raw_bias_weight_derivs)