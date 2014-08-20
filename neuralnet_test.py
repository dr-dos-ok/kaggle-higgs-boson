import unittest
import numpy as np
from numpy.testing import *
from scipy.special import expit

import neuralnet as nn
import neurons

class TestFunctions(unittest.TestCase):
	def test_adjacent_pairs(self):
		pairs_list = [1,2,3,4]
		expected_pairs = [(1,2), (2,3), (3,4)]
		actual_pairs = [pair for pair in nn.adjacent_pairs(pairs_list)]
		assert_array_equal(expected_pairs, actual_pairs)

	def test_logistic(self):
		inputs = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
		expected = 1.0 / (1.0 + np.exp(-inputs))
		actual = nn.logistic(inputs)
		assert_array_equal(expected, actual)

	def test_logistic_deriv(self):
		inputs = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])
		log = 1.0 / (1.0 + np.exp(-inputs))
		expected = log * (1.0 - log)
		actual = nn.logistic_deriv(inputs)
		assert_array_equal(expected, actual)

	def test_flatten_lists_of_arrays(self):
		l1 = [
			np.array([1,2,3]),
			np.array([
				[4,5],
				[6,7]
			]),
			np.array([8,9,10])
		]
		l2 = [
			np.array([10,20,30]),
			np.array([40,50,60,70]),
			np.array([80,90,100])
		]
		expected = np.array([1,2,3,4,5,6,7,8,9,10, 10,20,30,40,50,60,70,80,90,100])
		actual = nn.flatten_lists_of_arrays(l1, l2)
		assert_array_equal(actual, expected)

	def test_unflatten_to_lists_of_arrays(self):

		flattened = np.array([1,2,3,4,5,6,7,8,9,10, 10,20,30,40,50,60,70,80,90,100])

		expected1 = [
			np.array([1,2,3]),
			np.array([
				[4,5],
				[6,7]
			]),
			np.array([8,9,10])
		]

		expected2 = [
			np.array([10,20,30]),
			np.array([40,50,60,70]),
			np.array([80,90,100])
		]

		outlist1 = [
			np.zeros(3),
			np.zeros((2,2)),
			np.zeros(3)
		]
		outlist2 = [
			np.zeros(3),
			np.zeros(4),
			np.zeros(3)
		]

		nn.unflatten_to_lists_of_arrays(flattened, outlist1, outlist2)

		for index in range(len(expected1)):
			assert_array_equal(expected1[index], outlist1[index])
			assert_array_equal(expected2[index], outlist2[index])

class TestFeedForwardNet(unittest.TestCase):

	def test_forward(self):
		net = nn.FeedForwardNet([5, 3, 1])
		weights = net.weights = [
			1.0 / np.array([
				[1.0, 2.0, 3.0],
				[4.0, 5.0, 6.0],
				[7.0, 8.0, 9.0],
				[10.0, 11.0, 12.0],
				[13.0, 14.0, 15.0]
			]),
			1.0 / np.array([
				[1.0],
				[2.0],
				[3.0]
			])
		]
		bias_weights = net.bias_weights = [
			None,
			np.array([16.0, 17.0, 18.0]),
			np.array([4.0])
		]

		inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

		expected_activations = [
			np.empty(0),
			np.dot(inputs.transpose(), weights[0]) + bias_weights[1],
			np.dot(
				expit(np.dot(inputs.transpose(), weights[0]) + bias_weights[1]).transpose(),
				weights[1]
			) + bias_weights[2]
		]
		expected_outputs = [
			expit(layer_inputs) for layer_inputs in expected_activations
		]

		outputs, activations = net.forward(inputs, outputs=nn.DEBUGGING_OUTPUTS)

		assert_array_equal(
			inputs,
			outputs[0]
		)

		for i in range(1, len(weights)+1):
			assert_array_equal(
				expected_activations[i],
				activations[i]
			)
			assert_array_equal(
				expected_outputs[i],
				outputs[i]
			)

	def test_get_flattened_weights(self):
		net = nn.FeedForwardNet([2,3,2])
		net.weights = [
			np.array([
				[0.1, 0.2, 0.3],
				[-0.1, -0.2, -0.3]
			]),
			np.array([
				[0.5, 0.5],
				[0.5, -0.5],
				[-0.5, 0.5]
			])
		]

		net.bias_weights = [
			None,
			np.array([0.3, 0.2, 0.1]),
			np.array([-0.5, -0.5])
		]

		expected = np.array([
			0.1, 0.2, 0.3, -0.1, -0.2, -0.3,
			0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
			0.3, 0.2, 0.1, -0.5, -0.5
		])
		flattened = net.get_flattened_weights()
		assert_array_equal(flattened, expected)

	def test_set_flattened_weights(self):
		net = nn.FeedForwardNet([2,3,2])

		net.set_flattened_weights(np.array([
			0.1, 0.2, 0.3, -0.1, -0.2, -0.3,
			0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
			0.3, 0.2, 0.1, -0.5, -0.5
		]))

		expected_weights = [
			np.array([
				[0.1, 0.2, 0.3],
				[-0.1, -0.2, -0.3]
			]),
			np.array([
				[0.5, 0.5],
				[0.5, -0.5],
				[-0.5, 0.5]
			])
		]

		expected_bias_weights = [
			None,
			np.array([0.3, 0.2, 0.1]),
			np.array([-0.5, -0.5])
		]

		for index in range(len(expected_weights)):
			assert_array_equal(expected_weights[index], net.weights[index])

		for index in range(len(expected_bias_weights)):
			if expected_bias_weights[index] is None:
				self.assertEqual(expected_bias_weights[index], net.bias_weights[index])
			else:
				assert_array_equal(expected_bias_weights[index], net.bias_weights[index])

	def test_get_partial_derivs(self):
		layer_sizes = [2, 3, 2]
		weights = [
			np.array([
				[0.1, 0.2, 0.3],
				[-0.1, -0.2, -0.3]
			]),
			np.array([
				[0.5, 0.5],
				[0.5, -0.5],
				[-0.5, 0.5]
			])
		]
		bias_weights = [
			None,
			np.array([0.3, 0.2, 0.1]),
			np.array([-0.5, -0.5])
		]
		inputs = np.array([1.0, -1.0])
		expected_outputs = np.array([1.0, 0.0])

		###
		# Make validation net with neurons package
		###

		input_layer = [neurons.InputNeuron() for i in range(layer_sizes[0])]
		input_bias = neurons.BiasNeuron()
		hidden_layer = [neurons.LogisticNeuron() for i in range(layer_sizes[1])]
		hidden_bias = neurons.BiasNeuron()
		output_layer = [neurons.SquaredErrorOutputNeuron() for i in range(layer_sizes[2])]

		for hidden_index, hidden_neuron in enumerate(hidden_layer):
			for input_index, input_neuron in enumerate(input_layer):
				neurons.make_connection(input_neuron, hidden_neuron, weights[0][input_index][hidden_index])
			neurons.make_connection(input_bias, hidden_neuron, bias_weights[1][hidden_index])

		for output_index, output_neuron in enumerate(output_layer):
			for hidden_index, hidden_neuron in enumerate(hidden_layer):
				neurons.make_connection(hidden_neuron, output_neuron, weights[1][hidden_index][output_index])
			neurons.make_connection(hidden_bias, output_neuron, bias_weights[2][output_index])

		for index, input_neuron in enumerate(input_layer):
			input_neuron.set_value(inputs[index])

		#forward pass
		for hidden_neuron in hidden_layer:
			hidden_neuron.forward_pass()
		for output_neuron in output_layer:
			output_neuron.forward_pass()

		#backprop

		for output_index, output_neuron in enumerate(output_layer):
			output_neuron.set_expected_value(expected_outputs[output_index])
		for output_neuron in output_layer:
			output_neuron.backprop()
		for hidden_neuron in hidden_layer:
			hidden_neuron.backprop()
		hidden_bias.backprop()

		#calc weight partial derivatives
		for output_neuron in output_layer:
			for weight in output_neuron.lower_connections:
				weight.calc_derror_by_dweight()
		for hidden_neuron in hidden_layer:
			for weight in hidden_neuron.lower_connections:
				weight.calc_derror_by_dweight()

		#now go grab all of the calculated values
		expected_weight_derivs = [
			np.array([
				[weight.derror_by_dweight for weight in input_neuron.upper_connections]
				for input_neuron in input_layer
			]),
			np.array([
				[weight.derror_by_dweight for weight in hidden_neuron.upper_connections]
				for hidden_neuron in hidden_layer
			])
		]

		expected_bias_weight_derivs = [
			None,
			np.array([weight.derror_by_dweight for weight in input_bias.upper_connections]),
			np.array([weight.derror_by_dweight for weight in hidden_bias.upper_connections])
		]

		###
		# make net that we're actually going to test with neuralnet/nn package
		###
		net = nn.FeedForwardNet(layer_sizes)
		net.weights = weights
		net.bias_weights = bias_weights

		#forward & backward pass all in one go
		weight_partial_derivs, bias_weight_partial_derivs = net.get_partial_derivs(inputs, expected_outputs)

		assert_almost_equal(
			expected_weight_derivs[0],
			weight_partial_derivs[0]
		)
		assert_almost_equal(
			expected_weight_derivs[1],
			weight_partial_derivs[1]
		)

		#both should be None
		self.assertEqual(expected_bias_weight_derivs[0], bias_weight_partial_derivs[0])

		assert_almost_equal(
			expected_bias_weight_derivs[1],
			bias_weight_partial_derivs[1]
		)
		assert_almost_equal(
			expected_bias_weight_derivs[2],
			bias_weight_partial_derivs[2]
		)

class TestNeurons(unittest.TestCase):
	def test_neuron(self):
		"""same net as TestFeedForwardNet.test_forward()"""

		input_layer = [neurons.InputNeuron() for i in range(5)]
		input_bias = neurons.BiasNeuron()
		hidden_layer = [neurons.LogisticNeuron() for i in range(3)]
		hidden_bias = neurons.BiasNeuron()
		output_layer = [neurons.SquaredErrorOutputNeuron() for i in range(1)]

		weights = [
			1.0 / np.array([
				[1.0, 2.0, 3.0],
				[4.0, 5.0, 6.0],
				[7.0, 8.0, 9.0],
				[10.0, 11.0, 12.0],
				[13.0, 14.0, 15.0],
				[16.0, 17.0, 18.0]
			]),
			1.0 / np.array([
				[1.0],
				[2.0],
				[3.0],
				[4.0]
			])
		]

		for hidden_index, hidden_neuron in enumerate(hidden_layer):
			for input_index, input_neuron in enumerate(input_layer):
				neurons.make_connection(input_neuron, hidden_neuron, weights[0][input_index][hidden_index])
			neurons.make_connection(input_bias, hidden_neuron, weights[0][-1][hidden_index])

		for output_index, output_neuron in enumerate(output_layer):
			for hidden_index, hidden_neuron in enumerate(hidden_layer):
				neurons.make_connection(hidden_neuron, output_neuron, weights[1][hidden_index][output_index])
			neurons.make_connection(hidden_bias, output_neuron, weights[1][-1][output_index])

		inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		inputs_with_bias = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])
		for input_index, input_neuron in enumerate(input_layer):
			input_neuron.set_value(inputs[input_index])

		#forward pass
		for hidden_neuron in hidden_layer:
			hidden_neuron.forward_pass()
		for output_neuron in output_layer:
			output_neuron.forward_pass()


		expected_activations = [
			np.empty(6),
			np.dot(inputs_with_bias.transpose(), weights[0]),
			np.dot(
				np.concatenate([
					expit(np.dot(inputs_with_bias.transpose(), weights[0])),
					np.ones(1)
				]).transpose(),
				weights[1]
			)
		]
		expected_outputs = [
			inputs_with_bias,
			expit(np.dot(inputs_with_bias.transpose(), weights[0])),
			expit(
				np.dot(
					np.concatenate([
						expit(np.dot(inputs_with_bias.transpose(), weights[0])),
						np.ones(1)
					]).transpose(),
					weights[1]
				)
			)
		]

		assert_array_equal(
			expected_activations[1],
			[neuron.input for neuron in hidden_layer]
		)
		assert_array_equal(
			expected_activations[2],
			[neuron.input for neuron in output_layer]
		)
		assert_array_equal(
			expected_outputs[1],
			[neuron.output for neuron in hidden_layer]
		)
		assert_array_equal(
			expected_outputs[2],
			[neuron.output for neuron in output_layer]
		)

if __name__ == "__main__":
	unittest.main()