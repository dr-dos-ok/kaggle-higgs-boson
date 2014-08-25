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
		self.derror_by_dinput = 1.0 - (self.output * self.output) * self.derror_by_doutput

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

class SquaredErrorOutputNeuron(LogisticNeuron):
	def __init__(self):
		super(SquaredErrorOutputNeuron, self).__init__()
		self.upper_connections = None

	def set_expected_value(self, value):
		self.expected_value = value

	#override
	def calc_derror_by_doutput(self):
		self.derror_by_doutput = self.output - self.expected_value

	#override
	def add_upper_connection(self):
		raise Exception("You probably shouldn't do this from an output neuron")

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