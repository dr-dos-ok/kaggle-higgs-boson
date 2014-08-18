import unittest
import numpy as np
from numpy.testing import *
from scipy.special import expit

import neuralnet as nn

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
			np.concatenate([
				expit(np.dot(inputs_with_bias.transpose(), weights[0])),
				np.ones(1)
			]),
			np.concatenate([
				expit(
					np.dot(
						np.concatenate([
							expit(np.dot(inputs_with_bias.transpose(), weights[0])),
							np.ones(1)
						]).transpose(),
						weights[1]
					)
				),
				np.ones(1)
			])
		]

		outputs, outputs_all, activations = net.forward(inputs, return_all=True)

		assert_array_equal(
			inputs,
			outputs_all[0][:-1]
		)
		self.assertEqual(1.0, outputs_all[0][-1])

		for i in range(1, len(weights)+1):
			assert_array_equal(
				expected_activations[i],
				activations[i]
			)
			assert_array_equal(
				expected_outputs[i],
				outputs_all[i]
			)

if __name__ == "__main__":
	unittest.main()