import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt

import sys, time, bkputils

# def make_isnoise_cols(dataframe, feature_cols):
# 	isnoise_cols = ["ISNOISE" + col[3:] for col in feature_cols]
# 	for col, isnoise_col in zip(feature_cols, isnoise_cols):
# 		dataframe[isnoise_col] = np.empty(dataframe.shape[0], dtype=np.int) # we'll fill these later
# 	return isnull_cols

def rand_bits(shape):
	return np.random.randint(0, 2, size=shape)

def fill_columns(df, columns, fill_fn):
	for col in columns:
		df[col] = fill_fn(df.shape[0])

def pretrain(
	all_traindata, validation_ids, training_ids,
	feature_cols, isnull_cols, output_cols,
	layer_sizes,
	hidden_fn_pair, output_layer,
	learning_rate=0.01,
	velocity_decay=0.9,
	weight_decay=0.05,
	batch_size=100
):
	#don't mess with the original
	all_traindata = all_traindata.copy()

	isnoise_cols = ["ISNOISE" + col[3:] for col in feature_cols]
	fill_columns(all_traindata, isnoise_cols, np.ones)

	training_set = all_traindata.loc[training_ids]
	validation_set = all_traindata.loc[validation_ids]

	batch_cutoffs = np.concatenate([
		np.arange(0, training_set.shape[0], batch_size),
		np.array([training_set.shape[0]])
	])

	plotter = Plotter()

	all_inputs = feature_cols + isnull_cols + isnoise_cols
	all_outputs = output_cols + feature_cols

	hidden_layer_sizes = layer_sizes[1:-1] #chop off first and last
	shuffled_train_ids = training_set.index.values.copy() #we'll shuffle these repeatedly, later
	net = None
	best_weights = None
	best_error = np.inf
	num_epochs = 0
	bkputils.capture_sigint()

	#add each hidden layer one at a time
	for layer_index in range(len(hidden_layer_sizes)):

		current_layer_sizes = [len(all_inputs)] + hidden_layer_sizes[0:layer_index+1] + [len(all_outputs)]

		prev_net = net
		net = nn.FeedForwardNet(
			current_layer_sizes,
			hidden_fn_pair=hidden_fn_pair,
			output_layer=output_layer
		)

		#copy weights from previous net
		#note: will skip first (None) net b/c range(0)=[]
		for i in range(layer_index):
			net._weights[i][:] = prev_net._weights[i]
			net._bias_weights[i+1][:] = prev_net._bias_weights[i+1]

		layer_epochs = 0

		train_err = get_err(net, training_set[all_inputs].values, training_set[all_outputs].values)
		validation_err = get_err(net, validation_set[all_inputs].values, validation_set[all_outputs].values)
		plotter.plot(num_epochs, train_err, validation_err)

		weights = net.get_flattened_weights()
		best_weights = weights.copy()
		best_error = validation_err
		best_epoch = -1
		velocity = net.zeros_like_flattened_weights()

		#train new layer until ctrl-c is pressed
		# 1 loop = 1 epoch (pass thru data)
		# while not bkputils.is_cancelled():
		for i in range(1):

			np.random.shuffle(shuffled_train_ids)
			epoch_set = training_set.loc[shuffled_train_ids] #shuffled copy, changes should not affect training_set

			fill_columns(epoch_set, isnoise_cols, rand_bits)
			epoch_set[feature_cols] *= epoch_set[isnoise_cols].values
			
			epoch_set = epoch_set.loc[shuffled_train_ids]

			weights -= weights * weight_decay
			# velocity = net.zeros_like_flattened_weights()

			#do a bunch of minibatches per epoch
			for batch_num, (batch_start, batch_stop) in enumerate(nn.adjacent_pairs(batch_cutoffs)):

				batch = epoch_set.iloc[batch_start:batch_stop]

				gradient = net.get_partial_derivs(
					batch[all_inputs].values,
					batch[all_outputs].values,
					outputs=nn.FLATTENED_WEIGHTS
				)

				velocity *= velocity_decay
				velocity += -(gradient * learning_rate)
				weights += velocity
			#end minibatch loop

			layer_epochs += 1
			num_epochs += 1

			#get error with no noise (all noise columns should be 1, and
			#all data columns should be themselves)
			train_err = get_err(net, training_set[all_inputs].values, training_set[all_outputs].values)
			validation_err = get_err(net, validation_set[all_inputs].values, validation_set[all_outputs].values)
			plotter.plot(num_epochs, train_err, validation_err)

			if validation_err < best_error:
				best_error = validation_err
				best_weights = weights.copy()
				best_epoch = num_epochs
		#end train loop
		bkputils.reset_cancelled()

		# plt.close()
		print "layer {0} finished: err = {1}".format(layer_index+1, best_error)
	#end hidden layer loop
	bkputils.uncapture_sigint()

	net.set_flattened_weights(best_weights)
	return net

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