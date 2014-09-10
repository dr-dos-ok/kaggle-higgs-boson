import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

import sys, bkputils, time

WEIGHT_DECAY = 0.05

def train(
	net,
	all_traindata, validation_ids, training_ids,
	feature_cols, output_cols,
	learning_rate=0.01,
	velocity_decay=0.9,
	batch_size=100
):
	training_set = all_traindata.loc[training_ids]
	validation_set = all_traindata.loc[validation_ids]

	batch_cutoffs = np.concatenate([
		np.arange(0, training_set.shape[0], batch_size),
		np.array([training_set.shape[0]])
	])
	batches_per_epoch = batch_cutoffs.shape[0] - 1

	train_err = get_err(net, training_set[feature_cols].values, training_set[output_cols].values)
	validation_err = get_err(net, validation_set[feature_cols].values, validation_set[output_cols].values)

	plotter = Plotter()
	plotter.plot(0.0, train_err, validation_err)

	weights = net.get_flattened_weights()
	best_weights = weights.copy()
	best_error = validation_err
	best_epoch = -1
	velocity = net.zeros_like_flattened_weights()

	num_epochs = 0
	shuffled_indices = training_set.index.values.copy()
	bkputils.capture_sigint()
	while not bkputils.is_cancelled():

		np.random.shuffle(shuffled_indices)
		shuffled_training_set = training_set.loc[shuffled_indices]

		weights -= weights * WEIGHT_DECAY
		for batch_num, (batch_start, batch_stop) in enumerate(nn.adjacent_pairs(batch_cutoffs)):
			batch = shuffled_training_set.iloc[batch_start:batch_stop]

			gradient = net.get_partial_derivs(
				batch[feature_cols].values,
				batch[output_cols].values,
				outputs=nn.FLATTENED_WEIGHTS
			)

			velocity *= velocity_decay
			velocity += -(gradient * learning_rate)
			weights += velocity # modifying net's internals; no need to call setter or anything

		num_epochs += 1
		train_err = get_err(net, training_set[feature_cols].values, training_set[output_cols].values)
		validation_err = get_err(net, validation_set[feature_cols].values, validation_set[output_cols].values)
		plotter.plot(num_epochs, train_err, validation_err)

		if validation_err < best_error:
			best_error = validation_err
			best_weights = weights.copy()
			best_epoch = num_epochs

	bkputils.uncapture_sigint()

	net.set_flattened_weights(best_weights)
	return best_error

def get_err(net, inputs, targets):
	errs = net.err_fn(
		net.forward(inputs),
		targets
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