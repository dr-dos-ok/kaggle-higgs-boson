import numpy as np
import pandas as pd
import neuralnet as nn
import matplotlib.pyplot as plt

import math

EPOCHS_PER_DNA = 50
NOISINESS = 0.2

def pretrain2(
	all_traindata, validation_ids, train_ids,
	feature_cols,
	layer_sizes,
	hidden_fn_pair, output_layer,
	learning_rate=0.01,
	velocity_decay=0.9,
	weight_decay=0.05,
	batch_size=100
):

	validation_set = all_traindata.loc[validation_ids, feature_cols]
	train_set = all_traindata.loc[train_ids, feature_cols]

	working_validation_set = validation_set.values
	working_training_set = train_set.values

	plotter = Plotter()

	dnas = []
	working_data = all_traindata
	for layer_size in layer_sizes:
		dna = _make_dna(
			working_training_set,
			working_validation_set,
			layer_size,
			plotter,
			learning_rate, velocity_decay, weight_decay, batch_size
		)
		working_validation_set = dna.forward(working_validation_set)
		working_training_set = dna.forward(working_training_set)
		dnas.append(dna)

	#this isn't even really going to be a dna of the input,
	#I'm not sure what this would be called...
	result_dna = nn.FeedForwardNet(
		[len(feature_cols)] + layer_sizes,
		hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
		output_layer=nn.TANH_MOD_SQERR_OUTPUT_LAYER
	)
	for layer_index, layer_size in enumerate(layer_sizes):
		result_dna._weights[layer_index][:] = dnas[layer_index]._weights[0]
		result_dna._bias_weights[layer_index+1][:] = dnas[layer_index]._bias_weights[1]

	return result_dna

def _make_dna(
	train_set,
	validation_set,
	size,
	plotter,
	learning_rate, velocity_decay, weight_decay, batch_size
):
	dna = nn.FeedForwardNet(
		[train_set.shape[1], size, train_set.shape[1]],
		hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
		output_layer=nn.IDENTITY_SQERR_OUTPUT_LAYER
	)

	train_err = get_dna_err(dna, train_set)
	validation_err = get_dna_err(dna, validation_set)
	if np.isnan(train_err) or np.isnan(validation_err):
		print
		print validation_set
		exit()
	plotter.plot_another(train_err, validation_err)

	num_batches = int(math.ceil(train_set.shape[0] / batch_size))
	batch_cutoffs = np.linspace(0, train_set.shape[0], num=num_batches).astype(np.int)

	weights = dna.get_flattened_weights()
	best_weights = weights.copy()
	best_error = validation_err
	velocity = dna.zeros_like_flattened_weights()
	shuffled_indices = range(train_set.shape[0]) #train_set.index.values.copy()
	for num_epochs in range(EPOCHS_PER_DNA):

		np.random.shuffle(shuffled_indices)
		epoch_targets = train_set[shuffled_indices]

		# noise = np.random.randint(0, 2, size=epoch_targets.shape)
		noise = np.random.binomial(1, NOISINESS, size=epoch_targets.shape)
		epoch_inputs = epoch_targets * noise

		for batch_start, batch_stop in nn.adjacent_pairs(batch_cutoffs):

			gradient = dna.get_partial_derivs(
				epoch_inputs[batch_start:batch_stop],
				epoch_targets[batch_start:batch_stop],
				outputs=nn.FLATTENED_WEIGHTS
			)

			velocity *= velocity_decay
			velocity += -(gradient * learning_rate)
			weights += velocity

		train_err = get_dna_err(dna, train_set)
		validation_err = get_dna_err(dna, validation_set)
		plotter.plot_next(train_err, validation_err)

		if validation_err < best_error:
			best_error = validation_err
			best_weights = weights.copy()

	dna.set_flattened_weights(best_weights)

	#strip off the decoder layers; only keep input-hidden wieghts
	#and hidden biases
	result_dna = nn.FeedForwardNet(
		[train_set.shape[1], size],
		hidden_fn_pair = nn.TANH_MOD_FN_PAIR, #Note: this will not be used (no hidden layers)
		output_layer=nn.TANH_MOD_SQERR_OUTPUT_LAYER #Note: output layer now is hidden layer previously, so we have to match types
	)
	result_dna._weights[0][:] = dna._weights[0]
	result_dna._bias_weights[1][:] = dna._bias_weights[1]

	return result_dna

def get_dna_err(net, data):
	return get_err(net, data, data)

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

	def plot_next(self, train_err, validation_err):
		x = self.x_vals[-1] + 1
		self.plot(x, train_err, validation_err)

	def plot_another(self, train_err, validation_err):
		x = self.x_vals[-1]
		self.plot(x, train_err, validation_err)

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