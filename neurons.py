"""
Utility classes to implement a feedforward neural network.

These classes are not intended to do actual work; they will
be much too slow and memory intensive because each neuron and
each weight is implemented as its own class instance. Instead, these
classes are intended to compute values for various small test
cases, so that we can validate the results of the "real" 
implementation against these (the "real" implementation will
probably be the neuralnet.py package I'm currently implementing
in numpy).

Writing these classes should hopefully add more error-resistance
for 2 reasons:
	1 - I find it easier to think about the neuron and weight
	math on a one-by-one basis instead of formulating large
	chunks of math as a single matrix multiplication
	2 - It is unlikely that I would make the same implementation
	error in 2 different places, because these are 2 *very* different
	implementations. Any errors that occur in both implementations
	are likely due to my own misunderstandings and not due to
	random implementation bugs.
"""

import math

class SigmoidNeuron(object):
	def __init__(self):
		self.upper_connections = []
		self.lower_connections = []

	def add_upper_connection(self, connection):
		self.upper_connections.append(connection)

	def add_lower_connection(self, connection):
		self.lower_connections.append(connection)

	def calc_input(self):
		s = 0.0
		for connection in self.lower_connections:
			s += connection.weight * connection.lower_neuron.output
		self.input = s

	def calc_output(self):
		raise Exception("Please implement this in a proper subclass")

	def calc_derror_by_doutput(self):
		result = 0.0
		for connection in self.upper_connections:
			result += connection.weight * connection.upper_neuron.derror_by_dinput
		self.derror_by_doutput = result

	def calc_derror_by_dinput(self):
		raise Exception("Please implement this in a proper subclass")

	def forward_pass(self):
		self.calc_input()
		self.calc_output()

	def backprop(self):
		self.calc_derror_by_doutput()
		self.calc_derror_by_dinput()


class LogisticNeuron(SigmoidNeuron):
	#override
	def calc_output(self):
		self.output = 1.0 / (1.0 + math.exp(-self.input))

	#override
	def calc_derror_by_dinput(self):
		y = self.output
		self.derror_by_dinput = y * (1 - y) * self.derror_by_doutput

class TanhNeuron(SigmoidNeuron):
	#override
	def calc_output(self):
		self.output = math.tanh(self.input)

	#override
	def calc_derror_by_dinput(self):
		self.derror_by_dinput = (1.0 - (self.output * self.output)) * self.derror_by_doutput

class BiasNeuron(object):

	def __init__(self):
		self.output = 1.0
		self.upper_connections = []

	def add_upper_connection(self, connection):
		self.upper_connections.append(connection)

	def calc_derror_by_doutput(self):
		result = 0.0
		for connection in self.upper_connections:
			result += connection.weight * connection.upper_neuron.derror_by_dinput
		self.derror_by_doutput = result

	def forward_pass(self):
		pass

	def backprop(self):
		self.calc_derror_by_doutput()

class InputNeuron(object):
	def __init__(self, value=None):
		self.output = value
		self.upper_connections = []

	def add_upper_connection(self, connection):
		self.upper_connections.append(connection)

	def set_value(self, value):
		self.output = value

	def calc_derror_by_doutput(self):
		result = 0.0
		for connection in self.upper_connections:
			result += connection.weight * connection.upper_neuron.derror_by_dinput
		self.derror_by_doutput = result

class OutputNeuron(SigmoidNeuron):
	def __init__(self):
		super(OutputNeuron, self).__init__()
		self.upper_connections = None

	def set_expected_value(self, value):
		self.expected_value = value

	#override
	def add_upper_connection(self):
		raise Exception("You probably shouldn't do this from an output neuron")

class SquaredErrorOutputNeuron(OutputNeuron):
	#override
	def calc_output(self):
		#same as LogisticNeuron
		self.output = 1.0 / (1.0 + math.exp(-self.input))

	#override
	def calc_derror_by_dinput(self):
		#same as LogisticNeuron
		y = self.output
		self.derror_by_dinput = y * (1 - y) * self.derror_by_doutput

	#override
	def calc_derror_by_doutput(self):
		self.derror_by_doutput = self.output - self.expected_value

class SoftmaxOutputNeuron(OutputNeuron):
	def __init__(self, softmax_layer):
		super(SoftmaxOutputNeuron, self).__init__()
		self.softmax_layer = softmax_layer

	def calc_input(self):
		super(SoftmaxOutputNeuron, self).calc_input()
		self.exp = math.exp(self.input)

	#override
	def calc_output(self):
		self.output = self.exp / self.softmax_layer.partition_fn()

	#override
	def calc_derror_by_dinput(self):
		self.derror_by_dinput = self.output - self.expected_value

	#override
	def calc_derror_by_doutput(self):
		raise Exception("You should probably skip straight to calc_derror_by_dinput()")

class NeuronLayer(object):
	def __init__(self):
		self.bias_neuron = BiasNeuron()

	def forward_pass(self):
		for neuron in self.neurons:
			neuron.calc_input()
		for neuron in self.neurons:
			neuron.calc_output()

	def backprop(self):
		for neuron in self.neurons:
			neuron.calc_derror_by_doutput()
		for neuron in self.neurons:
			neuron.calc_derror_by_dinput()
		self.bias_neuron.calc_derror_by_doutput()

	def connect_above(self, lower_layer, weights, bias_weights):
		bias_neuron = self.bias_neuron
		for upper_index, upper_neuron in enumerate(self.neurons):
			for lower_index, lower_neuron in enumerate(lower_layer.neurons):
				make_connection(lower_neuron, upper_neuron, weights[lower_index][upper_index])
			make_connection(bias_neuron, upper_neuron, bias_weights[upper_index])

	def calc_input_weight_derivs(self):
		for neuron in self.neurons:
			for weight in neuron.lower_connections:
				weight.calc_derror_by_dweight()

	def get_output_weight_derivs(self):
		return [
			[weight.derror_by_dweight for weight in neuron.upper_connections]
			for neuron in self.neurons
		]

	def get_bias_weight_derivs(self):
		return [weight.derror_by_dweight for weight in self.bias_neuron.upper_connections]

class LogisticNeuronLayer(NeuronLayer):
	def __init__(self, how_many):
		super(LogisticNeuronLayer, self).__init__()
		self.neurons = [LogisticNeuron() for i in range(how_many)]

class TanhNeuronLayer(NeuronLayer):
	def __init__(self, how_many):
		super(TanhNeuronLayer, self).__init__()
		self.neurons = [TanhNeuron() for i in range(how_many)]

class InputNeuronLayer(NeuronLayer):
	def __init__(self, how_many):
		super(InputNeuronLayer, self).__init__()
		self.neurons = [InputNeuron() for i in range(how_many)]

	def set_inputs(self, inputs):
		for index, neuron in enumerate(self.neurons):
			neuron.set_value(inputs[index])

class OutputNeuronLayer(NeuronLayer):
	def set_expected_values(self, expected_values):
		for index, neuron in enumerate(self.neurons):
			neuron.set_expected_value(expected_values[index])

class SquaredErrorOutputNeuronLayer(OutputNeuronLayer):
	def __init__(self, how_many):
		super(SquaredErrorOutputNeuronLayer, self).__init__()
		self.neurons = [SquaredErrorOutputNeuron() for i in range(how_many)]

class SoftmaxOutputNeuronLayer(OutputNeuronLayer):
	def __init__(self, num_outputs):
		super(SoftmaxOutputNeuronLayer, self).__init__()
		self.neurons = [
			SoftmaxOutputNeuron(self) for i in range(num_outputs)
		]

	def partition_fn(self):
		return sum([neuron.exp for neuron in self.neurons])

	#override
	def backprop(self):
		for neuron in self.neurons:
			neuron.calc_derror_by_dinput()
		self.bias_neuron.calc_derror_by_doutput()

class NeuronConnection(object):
	def __init__(self, lower_neuron, upper_neuron, weight):
		self.lower_neuron = lower_neuron
		self.upper_neuron = upper_neuron
		self.weight = weight

	def calc_derror_by_dweight(self):
		self.derror_by_dweight = self.lower_neuron.output * self.upper_neuron.derror_by_dinput

def make_connection(lower_neuron, upper_neuron, weight):
	conn = NeuronConnection(lower_neuron, upper_neuron, weight)
	lower_neuron.add_upper_connection(conn)
	upper_neuron.add_lower_connection(conn)