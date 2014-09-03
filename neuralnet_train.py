import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

from neuralnet_classifier import ZNetClassifier
from bkputils import *

import sys, bkputils, time, cProfile

#1 epoch = 1 complete pass through data
MIN_EPOCHS = 10
# VALIDATION_PCT = 0.2
# BATCH_SIZE = 100
# VELOCITY_DECAY = 0.9
# LEARNING_RATE = 0.001

def train(net, alldata, learning_rate=0.01, velocity_decay=0.9, batch_size=100, validation_pct=0.2):
	
	write("initing validation and training sets")
	validation_size = int(alldata.shape[0] * validation_pct)
	index_shuffled = np.random.choice(alldata.index, alldata.shape[0])

	validation_indices = index_shuffled[:validation_size]
	training_indices = index_shuffled[validation_size:]

	validation_set = alldata.ix[validation_indices]
	training_set = alldata.ix[training_indices]
	writeDone()

	write("making batch_cutoffs")
	batch_cutoffs = np.concatenate([
		np.arange(0, training_set.shape[0], batch_size),
		np.array([training_set.shape[0]])
	])
	writeDone()

	write("making classifier")
	classifier = ZNetClassifier(net)
	writeDone()

	write("getting initial errs")
	train_err = get_err(net, training_set)
	validation_err = get_err(net, validation_set)
	writeDone()

	write("initing plotter")
	plotter = Plotter()
	plotter.plot(0.01,train_err, validation_err)
	writeDone()

	write("initing weights")
	weights = net.get_flattened_weights()
	best_weights = weights
	best_error = get_err(net, validation_set)
	best_epoch = 0
	velocity = net.zeros_like_flattened_weights()
	writeDone()

	overtrained = False
	num_epochs = 0
	bkputils.capture_sigint()
	while (
		((not overtrained) or (num_epochs < MIN_EPOCHS))
		and
		(not bkputils.is_cancelled())
	):

		write("shuffling indices")
		shuffled_indices = training_set.index.values.copy()
		np.random.shuffle(shuffled_indices)
		writeDone()

		write("minibatch training for a full epoch")
		#train epoch in batches of BATCH_SIZE
		getbatch = 0.0
		computegrad = 0.0
		weight_updates = 0.0
		set_weights = 0.0
		for batch_start, batch_stop in nn.adjacent_pairs(batch_cutoffs):

			a = time.time()

			batch = training_set.ix[shuffled_indices[batch_start:batch_stop]]

			b = time.time()

			gradient = None
			def foo():
				gradient = net.get_partial_derivs(
					batch,
					outputs=nn.FLATTENED_WEIGHTS
				)
			cProfile.runctx("foo()", globals(), locals(), "neuralnet_train.profile")
			exit()

			c = time.time()

			velocity *= velocity_decay
			velocity += -(gradient * learning_rate)
			weights += velocity

			d = time.time()

			# net.set_flattened_weights(weights)

			e = time.time()

			getbatch += (b - a)
			computegrad += (c - b)
			weight_updates += (d - c)
			set_weights += (e - d)
		writeDone()

		t = "\t\t\t"
		print t + "getbatch:", fmtTime(getbatch)
		print t + "computegrad:", fmtTime(computegrad)
		print t + "weight_updates:", fmtTime(weight_updates)
		print t + "set_weights:", fmtTime(set_weights)

		write("calcing and plotting error")
		#epoch complete: evaluate error
		num_epochs += 1
		train_err = get_err(net, training_set)
		validation_err = get_err(net, validation_set)
		plotter.plot(num_epochs, train_err, validation_err)
		writeDone()

		write("housekeeping")
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
		writeDone()

		exit()	

		
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