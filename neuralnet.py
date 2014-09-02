import numpy as np
import pandas as pd
import cPickle as pickle
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

def logistic_deriv(x, y):
	"""
	x: the actual x-values that we're dealing with (numpy.ndarray)
	y: sigmoid(x) (numpy.ndarray).
		This is just a convenience parameter; we could just calculate
		y ourselves from x, but we know it already will have been calculated
		by the time we get here so we may as well save some computation and
		just pass it. Useful because a lot of sigmoid derivatives (logistic, tanh)
		have very simple definitions that only depend on y

	If the logistic function is y = 1.0 / (1.0 + exp(x))
	then the derivative of that function can be written as:
	y' = y * (1 - y)
	"""
	return y * (1.0 - y)

LOGISTIC_FN_PAIR = (scipy.special.expit, logistic_deriv)

def tanh_deriv(x, y):
	"""
	x: the actual x-values that we're dealing with (numpy.ndarray)
	y: sigmoid(x) (numpy.ndarray).
		This is just a convenience parameter; we could just calculate
		y ourselves from x, but we know it already will have been calculated
		by the time we get here so we may as well save some computation and
		just pass it. Useful because a lot of sigmoid derivatives (logistic, tanh)
		have very simple definitions that only depend on y

	The derivative of y = tanh(x) is y' = 1.0 - y**2
	"""
	return 1.0 - (y * y)

TANH_FN_PAIR = (np.tanh, tanh_deriv)

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

LAST_LAYER_OUTPUTS = 0
ALL_LAYER_OUTPUTS = 1
ALL_LAYER_INPUTS_AND_OUTPUTS = 2

LISTS_OF_WEIGHTS = 0
FLATTENED_WEIGHTS = 1
ALL_DERIVS_AND_WEIGHTS = 2

_ONE = np.ones(1)

class FeedForwardNet(object):

	def __init__(
		self,
		layer_sizes,
		sigmoid_fn_pair=TANH_FN_PAIR,
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
		the same. Default is
		neuralnet.TANH_FN_PAIR (numpy.tanh, neuralnet.tanh_deriv)

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

	def forward(self, inputs, outputs=LAST_LAYER_OUTPUTS):
		"""
		Run the inputs through this FeedForward network in the standard
		forward direction.

		Setting outputs=LAST_LAYER_OUTPUTS (default) will return the outputs
		of the final layer of the network.

		Setting outputs=ALL_LAYER_OUTPUTS will return a list of ndarrays
		representing the outputs from each layer

		Setting outputs=ALL_LAYER_INPUTS_AND_OUTPUTS will return a tuple of
		(layer_inputs, layer_outputs) where layer_inputs is a list
		of numpy.ndarray representing the inputs of every neuron at
		every layer of the network. layer_outputs is similar, and of the
		same shape, but contains the outputs to each neuron in the net (after
		sigmoid activation)
		"""
		
		return_all_layer_inputs = (outputs == ALL_LAYER_INPUTS_AND_OUTPUTS)
		return_all_layer_outputs = (
			outputs == ALL_LAYER_INPUTS_AND_OUTPUTS or
			outputs == ALL_LAYER_OUTPUTS
		)

		if return_all_layer_inputs:
			all_layer_inputs = [None] * len(self.layer_sizes)
		if return_all_layer_outputs:
			all_layer_outputs = [None] * len(self.layer_sizes)

		layer_outputs = inputs
		if return_all_layer_outputs:
			all_layer_outputs[0] = layer_outputs

		for prev_layer_index, layer_index in adjacent_pairs(range(self.nlayers)):
			layer_inputs = np.dot(
				layer_outputs, # from prev layer
				self.weights[prev_layer_index]
			)
			layer_inputs += self.bias_weights[layer_index]
			if return_all_layer_inputs:
				all_layer_inputs[layer_index] = layer_inputs

			layer_outputs = self.sigmoid(layer_inputs)

			if return_all_layer_outputs:
				all_layer_outputs[layer_index] = layer_outputs

		if outputs == LAST_LAYER_OUTPUTS:
			return layer_outputs
		elif outputs == ALL_LAYER_OUTPUTS:
			return all_layer_outputs
		elif outputs == ALL_LAYER_INPUTS_AND_OUTPUTS:
			return (all_layer_inputs, all_layer_outputs)
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

	def get_partial_derivs(self, test_case_inputs, test_case_outputs, outputs=LISTS_OF_WEIGHTS):
		"""
		Given a matrix of training cases (rows are training cases, columns are features) and
		a matrix of expected output values for each training case (same matrix structure),
		compute the partial derivative of the error with respect to each weight and bias weight
		for each training case. Return the average partial derivatives for each weight and bias
		in the network.

		Internally this method implements the backpropagation algorithm to calculate the
		partial error derivatives.
		http://en.wikipedia.org/wiki/Backpropagation
		https://class.coursera.org/neuralnets-2012-001/lecture/39

		If outputs=LISTS_OF_WEIGHTS (default) is specified, a tuple of
		(weight_derivs, bias_weight_derivs) is returned where each item is a list of
		numpy.ndarray. The shape and order of weight_derivs will be identical to that
		of self.weights and similarly bias_weight_derivs will match up to self.bias_weights.
		Note that, like self.bias_weights, the first element of bias_weight_derivs will
		be None because there are no biases computed for the input layer.

		If outputs=FLATTENED_WEIGHTS is specified, the outputs will be a 1D
		numpy.ndarray that is compatible with the 1D arrays used by the other
		flattened_weights methods available in this class

		If outputs=ALL_DERIVS_AND_WEIGHTS is specified a tuple of
		(layer_input_derivs, layer_output_derivs, weight_derivs, bias_weight_derivs)
		is returned. This "numeric diarrhea mode" is mostly only useful for unit tests
		and other debugging purposes.
		"""

		if test_case_inputs.ndim == 1:
			test_case_inputs = test_case_inputs.reshape(1,-1)

		#memory-saving hack: we should really pass outputs=ALL_LAYER_INPUTS_AND_OUTPUTS
		#and then pass layer_inputs to self.sigmoid_deriv as well, but none of the
		#currently implemented sigmoids (tanh, logistic) require the x values so
		#we should save some memory by not even retrieving them
		layer_outputs = self.forward(test_case_inputs, outputs=ALL_LAYER_OUTPUTS)

		#init python arrays to appropriate length
		#we'll fill with ndarrays presently
		layer_output_derivs = [None] * len(self.layer_sizes)
		layer_input_derivs = [None] * len(self.layer_sizes)

		err, derror_by_doutput = self.err_fn(layer_outputs[-1], test_case_outputs)
		layer_output_derivs[-1] = derror_by_doutput

		for layer_index in range(self.nlayers-1, 0, -1):
			#memory-saving hack: none of the sigmoids currently in use require
			#the input (x) values. Only the output (y) values are needed, so 
			#don't even bother keeping the inputs in memory (can be large)
			layer_input_derivs[layer_index] = \
				self.sigmoid_deriv(None, layer_outputs[layer_index]) \
				* layer_output_derivs[layer_index]

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

		if outputs == LISTS_OF_WEIGHTS:
			return (weight_derivs, [None] + raw_bias_weight_derivs)
		elif outputs == FLATTENED_WEIGHTS:
			return flatten_lists_of_arrays(weight_derivs, raw_bias_weight_derivs)
		elif outputs == ALL_DERIVS_AND_WEIGHTS:
			return (layer_input_derivs, layer_output_derivs, weight_derivs, [None] + raw_bias_weight_derivs)

	def save(self, filename):
		with open(filename, "wb") as f:
			pickle.dump(self, f)

	@staticmethod
	def load(filename):
		with open(filename, "rb") as f:
			return pickle.load(f)

class ZFeedForwardNet(FeedForwardNet):
	"""
	Subclass of FeedForwardNet that automatically transforms columns to zscores before
	running the net.

	Instead of expecting a numpy.ndarray, this class generally expects a pandas.DataFrame
	as an input to all of its methods. It will convert to numpy.ndarray before delegating
	to superclass methods.

	Internally this class will filter out any columns that have a standard deviation of 0.0
	so as to avoid nan's when calculating the z-score. It will re-calculate the number of 
	input columns if necessary. This should all be transparent to external classes, who
	can pass 0-stddev columns without worry.
	"""

	# IMPLEMENTATION NOTE:
	# There is no need to override get get_partial_derivs() in this subclass. The method
	# get_partial_derivs() uses forward() in such a way that overriding forward() is
	# sufficient for the functionality we need.

	def __init__(
		self,
		dataframe,
		available_cols,
		layer_sizes,
		sigmoid_fn_pair=TANH_FN_PAIR,
		err_fn=squared_error
	):
		subdf = dataframe[available_cols]
		self.stddevs = subdf.std()
		self.means = subdf.mean()

		#filter columns that have no variance; causes nan's later on when we
		#divide
		available_cols = [col for col in available_cols if self.stddevs[col] != 0.0]
		self.stddevs = self.stddevs[available_cols]
		self.means = self.means[available_cols]
		self.available_cols = available_cols

		#re-calculate input layer; may have changed
		layer_sizes[0] = len(available_cols)

		super(ZFeedForwardNet, self).__init__(layer_sizes, sigmoid_fn_pair, err_fn)

	def to_zscores(self, dataframe):
		#expect pandas.DataFrame, return numpy.ndarray
		return ((dataframe[self.available_cols] - self.means) / self.stddevs).values

	#override
	def forward(self, inputs, outputs=LAST_LAYER_OUTPUTS):
		inputs = self.to_zscores(inputs)
		return super(ZFeedForwardNet, self).forward(inputs, outputs)