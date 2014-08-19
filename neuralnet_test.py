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

class TestFeedForwardNet(unittest.TestCase):

	def test_forward(self):
		net = nn.FeedForwardNet([5, 3, 1])
		weights = net.weights = [
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

		inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		inputs_with_bias = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])

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

	def test_get_partial_derivs(self):
		net = nn.FeedForwardNet([2, 3, 2])
		weights = net.weights = [
			np.array([
				[0.1, 0.2],
				[-0.1, -0.2],
				[0.2, 0.1],
				[0.0, 0.0]
			])
		]

class TestNeuron(unittest.TestCase):
	def test_neuron(self):
		"""same net as TestFeedForwardNet.test_forward()"""

		input_layer = [neurons.InputNeuron() for i in range(5)]
		input_bias = neurons.BiasNeuron()
		hidden_layer = [neurons.LogisticNeuron() for i in range(3)]
		hidden_bias = neurons.BiasNeuron()
		output_layer = [neurons.OutputNeuron() for i in range(1)]

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