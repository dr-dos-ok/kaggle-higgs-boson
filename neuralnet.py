import numpy as np
import pandas as pd
import cPickle as pickle
import math

from scipy.special import expit

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

def squared_error(y, target):
	a = (y - target)
	return 0.5 * a * a

def squared_error_deriv(y, target):
	return y - target

def cross_entropy_error(y, target):
	return -np.sum(
		target * np.log(y),
		axis=1
	)

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

LOGISTIC_FN_PAIR = (expit, logistic_deriv)

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

def tanh_mod(x):
	"""
	Slightly modified version of tanh per Yann LeCunn's suggestion in
	"Efficient Backprop"
	"""
	return 1.7159 * np.tanh(0.6666 * x)

def tanh_mod_deriv(x, y):
	"""
	derivative of tanh_mod
	"""
	numerator1 = (1.0 - y*y)
	numerator2 = (1.0 - 1.7159*1.7159)
	denominator = 1.7159

	return 0.6666 * ((numerator1 - numerator2) / denominator)

TANH_MOD_FN_PAIR = (tanh_mod, tanh_mod_deriv)

def softmax(x):
	e = np.exp(x)
	return e / np.sum(e, axis=1).reshape((-1, 1))

def logistic_sqerr_input_deriv(x, y, target):
	return logistic_deriv(x, y) * squared_error_deriv(y, target)

def tanh_sqerr_input_deriv(x, y, target):
	return tanh_deriv(x, y) * squared_error_deriv(y, target)

def tanh_mod_sqerr_input_deriv(x, y, target):
	return tanh_mod_deriv(x, y) * squared_error_deriv(y, target)

def softmax_cross_entropy_input_deriv(x, y, target):
	return y - target

LOGISTIC_SQERR_OUTPUT_LAYER = (expit, squared_error, logistic_sqerr_input_deriv)
TANH_SQERR_OUTPUT_LAYER = (np.tanh, squared_error, tanh_sqerr_input_deriv)
TANH_MOD_SQERR_OUTPUT_LAYER = (tanh_mod, squared_error, tanh_mod_sqerr_input_deriv)
SOFTMAX_CROSS_ENTROPY_OUTPUT_LAYER = (softmax, cross_entropy_error, softmax_cross_entropy_input_deriv)

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

def unflatten_weights(flattened_weights, layer_sizes):
	weights = []
	bias_weights = [None] # None signifies that first layer has no biases

	start = 0
	for bottom, top in adjacent_pairs(layer_sizes):
		size = bottom * top
		stop = start + size
		weight_matrix = flattened_weights[start:stop].reshape((bottom, top))
		weights.append(weight_matrix)
		start = stop

	for layer_size in layer_sizes[1:]: # skip first layer; no biases
		stop = start + layer_size
		bias_matrix = flattened_weights[start:stop]
		bias_weights.append(bias_matrix)
		start = stop

	return (weights, bias_weights)

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

#for flattened_weight_mask
WEIGHTS = 1
BIASES = 2

_ONE = np.ones(1)

class FeedForwardNet(object):

	def __init__(
		self,
		layer_sizes,
		hidden_fn_pair=TANH_FN_PAIR,
		output_layer=LOGISTIC_SQERR_OUTPUT_LAYER
	):
		"""
		layer_sizes: (required) a list of ints specifying the number of
		neurons in each layer of this network. There must be at least
		2 ints: the input and output layers respectively. Any additional
		layers will be hidden layers.

		TODO - parameters have changed, update comments
		"""

		self.layer_sizes = np.array(layer_sizes)
		self.nlayers = nlayers = len(layer_sizes)

		self.nweights = sum([bottom * top for bottom, top in adjacent_pairs(layer_sizes)])
		self.nbiases = sum(layer_sizes[1:])
		self.n_weights_and_biases = self.nweights + self.nbiases
		
		#create the "one ndarray to rule them all" and then slice it up into
		#individual weight matrices that can actually be used
		self.flattened_weights = self.empty_like_flattened_weights()
		self._weights, self._bias_weights = unflatten_weights(self.flattened_weights, self.layer_sizes)

		#init weight matrices with small random weights (be sure to use [:] to copy
		#values instead of creating a reference to a new matrix)
		for index, (bottom, top) in enumerate(adjacent_pairs(layer_sizes)):
			self._weights[index][:] = (2.0 * (np.random.random((bottom, top)) - 0.5)) / math.sqrt(bottom+1)
			self._bias_weights[index+1][:] = (2.0 * (np.random.random(top) - 0.5)) / math.sqrt(bottom+1)

		hidden_fn, hidden_deriv = hidden_fn_pair
		output_fn, self.err_fn, self.output_x_deriv = output_layer

		self.layer_sigmoids = [None] + ([hidden_fn] * (nlayers-2)) + [output_fn]
		self.layer_sigmoid_derivs = [None] + ([hidden_deriv] * (nlayers-2)) + [None]

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
				self._weights[prev_layer_index]
			)
			layer_inputs += self._bias_weights[layer_index]
			if return_all_layer_inputs:
				all_layer_inputs[layer_index] = layer_inputs

			layer_outputs = self.layer_sigmoids[layer_index](layer_inputs)

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

	def set_weights(self, new_weights, new_bias_weights):
		for _weight, weight in zip(self._weights, new_weights):
			_weight[:] = weight
		for _bias_weight, bias_weight in zip(self._bias_weights[1:], new_bias_weights[1:]):
			_bias_weight[:] = bias_weight

	def get_flattened_weights(self):
		"""
		Return a 1D numpy.ndarray that represents the flattened weights of this
		network.

		The structure of the flattened array is not specified to outside code, except
		to guarantee that it is consistent with other flattened_weights methods, and
		that you can apply element-wise operations in a meaningful way (e.g. you can
		add two flattened weight arrays to get the sum of the weights)
		"""
		# return flatten_lists_of_arrays(self.weights, self.bias_weights[1:])
		return self.flattened_weights

	def set_flattened_weights(self, flattened_weights):
		"""
		Set the weights of this network according to the passed flattened_weights

		The structure of the flattened array is not specified to outside code, except
		to guarantee that it is consistent with other flattened_weights methods, and
		that you can apply element-wise operations in a meaningful way (e.g. you can
		add two flattened weight arrays to get the sum of the weights)
		"""
		self.flattened_weights[:] = flattened_weights
		# unflatten_to_lists_of_arrays(flattened_weights, self.weights, self.bias_weights[1:])

	def zeros_like_flattened_weights(self):
		"""
		Return a 1D numpy.zeros array that has the same shape as the arrays used
		in other flattened_weights methods.

		The structure of the flattened array is not specified to outside code, except
		to guarantee that it is consistent with other flattened_weights methods, and
		that you can apply element-wise operations in a meaningful way (e.g. you can
		add two flattened weight arrays to get the sum of the weights)
		"""
		return np.zeros(self.n_weights_and_biases)

	def empty_like_flattened_weights(self):
		return np.empty(self.n_weights_and_biases)

	def ones_like_flattened_weights(self):
		return np.ones(self.n_weights_and_biases)

	def flattened_weight_mask(self, mask):
		include_weights = (mask & WEIGHTS) > 0
		include_biases = (mask & BIASES) > 0

		result = np.zeros(self.n_weights_and_biases)
		if include_weights:
			result[:self.nweights] = 1
		if include_biases:
			result[self.nweights:] = 1
		return result

	def get_partial_derivs(self, test_case_inputs, test_case_targets, outputs=LISTS_OF_WEIGHTS):
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
		layer_output_derivs = [None] * (len(self.layer_sizes))
		layer_input_derivs = [None] * len(self.layer_sizes)

		layer_input_derivs[-1] = self.output_x_deriv(None, layer_outputs[-1], test_case_targets)

		#all layer indexes except the first and last (input & output)
		layer_indexes = range(1, self.nlayers-1)

		for layer_index in reversed(layer_indexes):
			layer_output_derivs[layer_index] = np.dot(
				layer_input_derivs[layer_index+1],
				self._weights[layer_index].T
			)

			layer_input_derivs[layer_index] = (
				self.layer_sigmoid_derivs[layer_index](None, layer_outputs[layer_index]) *
				layer_output_derivs[layer_index]
			)

		flattened_weight_derivs = self.empty_like_flattened_weights()
		weight_derivs, bias_weight_derivs = unflatten_weights(flattened_weight_derivs, self.layer_sizes)

		for index in xrange(len(self._weights)):
			weight_derivs[index][:] = np.dot(layer_outputs[index].T, layer_input_derivs[index+1]) / layer_outputs[index].shape[0]

		for index in xrange(len(self._bias_weights[1:])):
			index += 1 # because we skipped the first layer
			bias_weight_derivs[index][:] = np.mean(layer_input_derivs[index], axis=0)

		if outputs == LISTS_OF_WEIGHTS:
			return (weight_derivs, bias_weight_derivs)
		elif outputs == FLATTENED_WEIGHTS:
			return flattened_weight_derivs
		elif outputs == ALL_DERIVS_AND_WEIGHTS:
			return (layer_input_derivs, layer_output_derivs, weight_derivs, bias_weight_derivs)

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
		input_cols,
		output_cols,
		layer_sizes,
		hidden_fn_pair=TANH_FN_PAIR,
		output_layer=LOGISTIC_SQERR_OUTPUT_LAYER
	):
		#take a little extra care to filter out nans and -999 before computing
		#means/stddevs. Having nans will cause a nansplosion later on,
		#and having -999s will just skew the averages and stddevs a bit
		self.stddevs = pd.Series()
		self.means = pd.Series()
		_input_cols = []
		for col in input_cols:
			column = dataframe[col]
			column = column[(~np.isnan(column)) & (column != -999.0)]
			stddev = column.std()
			if stddev > 0.0:
				self.stddevs[col] = stddev
				self.means[col] = column.mean()
				_input_cols.append(col)
		self.input_cols = input_cols = _input_cols
		self.output_cols = output_cols

		#re-calculate input/output layer sizes; may have changed
		layer_sizes[0] = len(input_cols)
		layer_sizes[-1] = len(output_cols)

		super(ZFeedForwardNet, self).__init__(
			layer_sizes,
			hidden_fn_pair=hidden_fn_pair,
			output_layer=output_layer
		)

	def to_zscores(self, dataframe):

		dataframe = dataframe.copy()
		for col in self.input_cols:
			dataframe.loc[dataframe[col]==-999, col] = self.means[col]

		#expect pandas.DataFrame, return numpy.ndarray
		zscores = ((dataframe[self.input_cols] - self.means) / self.stddevs).values

		return zscores

	#override
	def forward(self, inputs, outputs=LAST_LAYER_OUTPUTS):
		inputs = self.to_zscores(inputs)
		return super(ZFeedForwardNet, self).forward(inputs, outputs)

	def get_partial_derivs(self, dataframe, outputs=LISTS_OF_WEIGHTS):
		return super(ZFeedForwardNet, self).get_partial_derivs(
			dataframe[self.input_cols],
			dataframe[self.output_cols],
			outputs=outputs
		)