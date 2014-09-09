import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

from bkputils import *

import sys, time

def train(net, validation_set, training_set, learning_rate=0.01, velocity_decay=0.9, batch_size=100):

	batch_cutoffs = np.concatenate([
		np.arange(0, training_set.shape[0], batch_size),
		np.array([training_set.shape[0]])
	])

	train_err = get_err(net, training_set)
	validation_err = get_err(net, validation_set)

def get_err(net, set):
	errs = net.err_fn(
		net.forward(df),
		df
	)

class Plotter(object):
	def __init__(self):
		self.x_vals = [0.0]
		self.train_errs = [0.0]
		self.validation_errs = [0.0]

		self.intermediate_x = [0.0]
		self.intermediate_errs = [0.0]

		plt.ion()
		self.train_line, self.validation_line, self.intermediate_line = plt.plot(
			self.x_vals, self.train_errs, "r-",
			self.x_vals, self.validation_errs, "b-",
			self.intermediate_x, self.intermediate_errs, "m."
		)
		plt.show()

	def plot(self, x, train_err, validation_err):
		self.x_vals.append(x)
		self.train_errs.append(train_err)
		self.validation_errs.append(validation_err)

		self._plot()

	def plot_intermediate(self, x, err):
		self.intermediate_x.append(x)
		self.intermediate_errs.append(err)

		self._plot()

	def _plot(self):
		self.train_line.set_xdata(self.x_vals)
		self.train_line.set_ydata(self.train_errs)

		self.validation_line.set_xdata(self.x_vals)
		self.validation_line.set_ydata(self.validation_errs)

		self.intermediate_line.set_xdata(self.intermediate_x)
		self.intermediate_line.set_ydata(self.intermediate_errs)

		plt.axes().relim()
		plt.axes().autoscale_view()

		plt.draw()