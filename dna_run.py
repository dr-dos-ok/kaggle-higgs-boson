import pandas as pd
import numpy as np
import neuralnet as nn
import sqlite3 as sqlite

from bkputils import *
from dna_train import pretrain

import os, zipfile, split

NET_SAVE_FILE = "dna_run.nn"
CSV_OUTPUT_FILE = "dna.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

def make_isnull_cols(dataframe):
	global feature_cols
	null_cols = [
		"PRI_jet_subleading_pt",
		"PRI_jet_leading_pt",
		"DER_mass_MMC"
	]
	isnull_cols = ["ISNULL" + col[3:] for col in null_cols]
	for col, isnull_col in zip(null_cols, isnull_cols):
		dataframe[isnull_col] = (dataframe[col] == -999.0).astype(np.int8)
	return isnull_cols

def make_trainfeature_cols(dataframe):
	dataframe["train_s"] = (dataframe["Label"] == "s").astype(np.int)
	dataframe["train_b"] = (dataframe["Label"] == "b").astype(np.int)

def split_output_vector(vector):
	"""
	The first two columns of the output will be a guess at the classes ("b" or "s", eventually)
	The rest we will use to try to reconstruct the (noisy) inputs to make this a de-noising
	autoencoder. We will use different activations and error functions for both sets.
	"""
	classes = vector[:,:2]
	inputs = vector[:,2:]
	return classes, inputs

def custom_output(x):
	x_classes, x_inputs = split_output_vector(x)
	result = np.hstack((
		nn.softmax(x_classes),
		x_inputs # identity function for these columns (no sigmoid)
	))
	return result

def custom_error(y, target):
	y_classes, y_inputs = split_output_vector(y)
	target_classes, target_inputs = split_output_vector(target)

	class_error = nn.cross_entropy_error(y_classes, target_classes)
	input_error = nn.squared_error(y_inputs, target_inputs)

	return class_error + input_error

def custom_error_input_deriv(x, y, target):
	y_classes, y_inputs = split_output_vector(y)
	target_classes, target_inputs = split_output_vector(target)

	#note: don't need x here, just pass 'None'
	class_deriv = nn.softmax_cross_entropy_input_deriv(None, y_classes, target_classes)

	#NOTE: we don't usually use squared_error_deriv directly; usually we would use
	#something like tanh_mod_sqerr_input_deriv which also multiplies by the
	#derivative of our chosen sigmoid function. However, since we're not using a sigmoid
	#for the inputs we don't do that here (derivative of the identity function is 1)
	input_deriv = nn.squared_error_deriv(y_inputs, target_inputs)

	return np.hstack((class_deriv, input_deriv))

CUSTOM_OUTPUT_LAYER = (custom_output, custom_error, custom_error_input_deriv)

write("loading training data")
conn = sqlite.connect("data.sqlite")
all_traindata = pd.read_sql("SELECT * FROM training", conn).set_index("EventId")
feature_cols = [col for col in all_traindata.columns.values if col.startswith("DER") or col.startswith("PRI")]
writeDone()

write("creating custom features on all_traindata")
make_trainfeature_cols(all_traindata)
isnull_cols = make_isnull_cols(all_traindata)
output_cols = ["train_s", "train_b"]
writeDone()

write("computing zscores on all_traindata")
means = pd.Series()
stddevs = pd.Series()
for col in feature_cols:
	column = all_traindata[col]
	isna = np.isnan(column) | (column == -999.0)
	column = column[~isna]
	stddevs[col] = column.std()
	means[col] = column.mean()
	all_traindata.loc[isna, col] = means[col]
	all_traindata[col] = (all_traindata[col] - means[col]) / stddevs[col]
writeDone()

write("loading training/validation id splits")
validation_ids, training_ids = split.load_ids()
writeDone()

write("pre-training net as de-noising autoencoder")
print # evens up some of the output later
layer_sizes=[len(feature_cols), 50, 50, 50, len(output_cols)]
prenet = pretrain(
	all_traindata, validation_ids, training_ids,
	feature_cols, isnull_cols, output_cols,
	layer_sizes,
	hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
	output_layer=CUSTOM_OUTPUT_LAYER
)
writeDone()

write("converting pretraining net to final form")
net = nn.FeedForwardNet(
	layer_sizes,
	hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
	output_layer=nn.SOFTMAX_CROSS_ENTROPY_OUTPUT_LAYER
)
#first, copy middle weights (excluding first and last layer); they can be copied directly
for i in range(1, len(net._weights)-1):
	net._weights[i][:] = prenet._weights[i][:]
	net._bias_weights[i][:] = prenet._bias_weights[i][:]
#first layer: first len(feature_cols) weights should be starting weights
#the second len(feature_cols) weights should be added to the bias of the
#first hidden layer (because their neurons will basically always be on, now)
net._weights[0][:] = prenet._weights[0][:len(feature_cols)]
net._bias_weights[1] += np.sum(prenet._weights[0][len(feature_cols):], axis=0)
#last layer: just drop the columns we don't need
net._weights[-1][:] = prenet._weights[-1][:, :len(output_cols)]
net._bias_weights[-1][:] = prenet._bias_weights[-1][:len(output_cols)]
writeDone()

write("fine-tuning net with normal backprop")
writeDone()