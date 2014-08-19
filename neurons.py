"""
Utility classes to implement a feedforward neural network for testing purposes.
These classes should be easier to implement and check the validity of, but
will be too slow for practical machine learning.
"""

import math

class LogisticNeuron(object):
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
		self.output = 1.0 / (1.0 + math.exp(-self.input))

	def calc_derror_by_doutput(self):
		result = 0.0
		for connection in self.upper_connections:
			result += connection.weight * connection.upper_neuron.derror_by_dinput
		self.derror_by_doutput = result

	def calc_derror_by_dinput(self):
		y = self.output
		self.derror_by_dinput = y * (1 - y) * self.derror_by_doutput

	def forward_pass(self):
		self.calc_input()
		self.calc_output()

	def backprop(self):
		self.calc_derror_by_doutput()
		self.calc_derror_by_dinput()

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

class OutputNeuron(LogisticNeuron):
	pass

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

class SquaredErrorQuasiNeuron(object):
	def __init__(self, expected_value=None):
		self.lower_connection = None
		self.expected_value = expected_value

	def set_expected_value(self, value):
		self.expected_value = value

	def add_lower_connection(self, connection):
		if self.lower_connection is None:
			self.lower_connection = connection
		else:
			raise Exception("You can only add one lower connection to an Error Neuron")

	def calc_derror_by_dinput(self):
		input_ = self.lower_connection.lower_neuron.output
		self.derror_by_doutput = input_ - self.expected_value

	def backprop(self):
		self.calc_derror_by_dinput()

class NeuronConnection(object):
	def __init__(self, lower_neuron, upper_neuron, weight):
		self.lower_neuron = lower_neuron
		self.upper_neuron = upper_neuron
		self.weight = weight

def make_connection(lower_neuron, upper_neuron, weight):
	conn = NeuronConnection(lower_neuron, upper_neuron, weight)
	lower_neuron.add_upper_connection(conn)
	upper_neuron.add_lower_connection(conn)