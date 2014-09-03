import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

from neuralnet_classifier import ZNetClassifier

import sys, bkputils

#1 epoch = 1 complete pass through data
MIN_EPOCHS = 10
# VALIDATION_PCT = 0.2
# BATCH_SIZE = 100
# VELOCITY_DECAY = 0.9
# LEARNING_RATE = 0.001

def train(net, alldata, learning_rate=0.01, velocity_decay=0.9, batch_size=100, validation_pct=0.2):
	
	validation_size = int(alldata.shape[0] * validation_pct)
	index_shuffled = np.random.choice(alldata.index, alldata.shape[0])

	validation_indices = index_shuffled[:validation_size]
	training_indices = index_shuffled[validation_size:]

	validation_set = alldata.ix[validation_indices]
	training_set = alldata.ix[training_indices]

	batch_cutoffs = np.concatenate([
		np.arange(0, training_set.shape[0], batch_size),
		np.array([training_set.shape[0]])
	])

	classifier = ZNetClassifier(net)

	plotter = Plotter()
	plotter.plot(
		0.01,
		get_err(net, training_set),
		get_err(net, validation_set)
	)

	weights = net.get_flattened_weights()
	best_weights = weights
	best_error = get_err(net, validation_set)
	best_epoch = 0
	velocity = net.zeros_like_flattened_weights()

	overtrained = False
	num_epochs = 0
	bkputils.capture_sigint()
	while (
		((not overtrained) or (num_epochs < MIN_EPOCHS))
		and
		(not bkputils.is_cancelled())
	):

		#train epoch in batches of BATCH_SIZE
		for batch_start, batch_stop in nn.adjacent_pairs(batch_cutoffs):
			batch = training_set.iloc[batch_start:batch_stop]

			gradient = net.get_partial_derivs(
				batch,
				batch[["train_feature"]],
				outputs=nn.FLATTENED_WEIGHTS
			)

			velocity *= velocity_decay
			velocity += -(gradient * learning_rate)
			weights += velocity
			net.set_flattened_weights(weights)

		#epoch complete: evaluate error
		num_epochs += 1
		train_err = get_err(net, training_set)
		validation_err = get_err(net, validation_set)
		plotter.plot(num_epochs, train_err, validation_err)

		if validation_err < best_error:
			best_error = validation_err
			best_weights = weights.copy()
			best_epoch = num_epochs

		if (
			(validation_err > (2.0 * train_err) and num_epochs > (best_epoch + 10))
			or
			(num_epochs > (best_epoch + 100))
		):
			overtrained = True

		
	bkputils.uncapture_sigint()

	net.set_flattened_weights(best_weights)
	return best_error

def get_err(net, df):
	errs, __ = net.err_fn(
		net.forward(df),
		df[["train_s", "train_b"]].values
	)
	return np.mean(errs)

class Plotter(object):
	def __init__(self):
		self.x_vals = [0.0]
		self.train_errs = [0.0]
		self.validation_errs = [0.0]

		plt.ion()
		self.train_line, self.validation_line = plt.plot(self.x_vals, self.train_errs, "r-", self.x_vals, self.validation_errs, "b-")
		plt.show()

	def plot(self, x, train_err, validation_err):
		self.x_vals.append(x)
		self.train_errs.append(train_err)
		self.validation_errs.append(validation_err)

		self.train_line.set_xdata(self.x_vals)
		self.train_line.set_ydata(self.train_errs)

		self.validation_line.set_xdata(self.x_vals)
		self.validation_line.set_ydata(self.validation_errs)

		plt.axes().relim()
		plt.axes().autoscale_view()

		plt.draw()