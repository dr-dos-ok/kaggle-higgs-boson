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
	for col in null_cols:
		newname = "ISNULL" + col[3:]
		feature_cols.append(newname)
		dataframe[newname] = (dataframe[col] == -999.0).astype(np.int8)

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
	return np.hstack(
		nn.softmax(x_classes),
		x_inputs # identity function for these columns (no sigmoid)
	)

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

	return class_deriv + input_deriv

CUSTOM_OUTPUT_LAYER = (custom_output, custom_error, custom_error_input_deriv)

write("loading training data")
conn = sqlite.connect("data.sqlite")
traindata = pd.read_sql("SELECT * FROM training", conn).set_index("EventId")
feature_cols = [col for col in traindata.columns.values if col.startswith("DER") or col.startswith("PRI")]
writeDone()

write("creating custom features on traindata")
make_isnull_cols(traindata)
make_trainfeature_cols(traindata)
output_cols = ["train_s", "train_b"]
writeDone()

write("pre-training net as de-noising autoencoder")
net = nn.ZFeedForwardNet(
	traindata,
	feature_cols, feature_cols,
	[-1, 50, -1],
	hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
	output_layer=nn.CUSTOM_OUTPUT_LAYER
)
min_err = pretrain(
	net,
	training_set
)
writeDone()