import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

from bkputils import *

import sys, bkputils, time

#1 epoch = 1 complete pass through data
MIN_EPOCHS = 10
# VALIDATION_PCT = 0.2
# BATCH_SIZE = 100
# VELOCITY_DECAY = 0.9
# LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.05

def train(net, validation_set, training_set, learning_rate=0.01, velocity_decay=0.9, batch_size=100):

	batch_cutoffs = np.concatenate([
		np.arange(0, training_set.shape[0], batch_size),
		np.array([training_set.shape[0]])
	])
	batches_per_epoch = batch_cutoffs.shape[0] - 1

	train_err = get_err(net, training_set)
	validation_err = get_err(net, validation_set)

	plotter = Plotter()
	plotter.plot(0.01, train_err, validation_err)

	weights = net.get_flattened_weights()
	best_weights = weights.copy()
	best_error = validation_err
	best_epoch = 0
	velocity = net.zeros_like_flattened_weights()

	weight_mask = net.flattened_weight_mask(nn.WEIGHTS)
	weight_decay_mask = WEIGHT_DECAY * weight_mask

	overtrained = False
	num_epochs = 0
	bkputils.capture_sigint()
	while (
		((not overtrained) or (num_epochs < MIN_EPOCHS))
		and
		(not bkputils.is_cancelled())
	):

		shuffled_indices = training_set.index.values.copy()
		np.random.shuffle(shuffled_indices)
		shuffled_training_set = training_set.loc[shuffled_indices]

		#train epoch in batches of BATCH_SIZE
		weights -= weights * weight_decay_mask
		velocity = net.zeros_like_flattened_weights()
		for batch_num, (batch_start, batch_stop) in enumerate(nn.adjacent_pairs(batch_cutoffs)):

			batch = shuffled_training_set.iloc[batch_start:batch_stop]

			gradient = net.get_partial_derivs(
				batch,
				outputs=nn.FLATTENED_WEIGHTS
			)

			velocity *= velocity_decay
			velocity += -(gradient * learning_rate)
			weights += velocity # since this is a direct reference to the net's actual weights, there's no need to call a setter or copy them somewhere

		#epoch complete: evaluate error
		num_epochs += 1
		train_err = get_err(net, training_set)
		validation_err = get_err(net, validation_set)
		plotter.plot(num_epochs, train_err, validation_err)

		# print
		# print "train_err, validation_err:", train_err, validation_err
		# __ = raw_input("Enter to exit")
		# exit()

		if validation_err < best_error:
			best_error = validation_err
			best_weights = weights.copy()
			best_epoch = num_epochs

		if (
			(validation_err > (2.0 * train_err) and num_epochs > (best_epoch + 500))
			# or
			# (num_epochs > (best_epoch + 100))
		):
			overtrained = True
		
	bkputils.uncapture_sigint()

	net.set_flattened_weights(best_weights)
	return best_error

def get_err(net, df):
	errs = net.err_fn(
		net.forward(df),
		df[["train_s", "train_b"]].values
	)
	return np.mean(errs)

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
