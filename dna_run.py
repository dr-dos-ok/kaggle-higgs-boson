import pandas as pd
import numpy as np
import neuralnet as nn
import sqlite3 as sqlite

from bkputils import *
from dna_pretrain2 import pretrain2
from dna_train import train
from neuralnet_classifier import ZNetClassifier

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
		dataframe[isnull_col] = (dataframe[col] == -999.0).astype(np.int)
	return isnull_cols

def make_trainfeature_cols(dataframe):
	dataframe["train_s"] = (dataframe["Label"] == "s").astype(np.int)
	dataframe["train_b"] = (dataframe["Label"] == "b").astype(np.int)

write("loading training data")
conn = sqlite.connect("data.sqlite")
all_traindata = pd.read_sql("SELECT * FROM training", conn).set_index("EventId")
feature_cols = [col for col in all_traindata.columns.values if col.startswith("DER") or col.startswith("PRI")]
writeDone()

write("creating custom features on all_traindata")
make_trainfeature_cols(all_traindata)
isnull_cols = make_isnull_cols(all_traindata)
feature_cols += isnull_cols
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
layer_sizes=[len(feature_cols)] + ([500]*3) + [len(output_cols)]
prenet = pretrain2(
	all_traindata, validation_ids, training_ids,
	feature_cols,
	layer_sizes[1:-1], #only pass hidden layer sizes
	hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
	output_layer=nn.IDENTITY_SQERR_OUTPUT_LAYER
)
writeDone()

write("converting pretraining net to final form")
net = nn.FeedForwardNet(
	layer_sizes,
	hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
	output_layer=nn.SOFTMAX_CROSS_ENTROPY_OUTPUT_LAYER
)
for i in range(len(layer_sizes)-2):
	net._weights[i][:] = prenet._weights[i]
	net._bias_weights[i+1][:] = prenet._bias_weights[i+1]
writeDone()

write("fine-tuning net with normal backprop")
err = train(
	net,
	all_traindata, validation_ids, training_ids,
	feature_cols, output_cols,
	learning_rate=0.01,
	velocity_decay=0.95
)
writeDone()
print "\terr:", err

write("saving net")
net.save(NET_SAVE_FILE)
writeDone()

write("searching for best cutoff value")
classifier = ZNetClassifier(net)
best_cutoff = 0.0
best_ams = 0.0
validation_set = all_traindata.loc[validation_ids]
validation_set_input_values = validation_set[feature_cols].values
for cutoff in np.arange(0.25, 0.75, 0.01):
	classifier.cutoff = cutoff
	predictions, confidence = classifier.classify(validation_set_input_values)
	score = ams(predictions, validation_set)
	if score > best_ams:
		best_ams = score
		best_cutoff = cutoff
classifier.cutoff = best_cutoff
writeDone()
print "\t\tcutoff: {0}".format(classifier.cutoff)
print "\t\tpredicted ams: {0}".format(best_ams)

#don't bother classifying the test data if this was a crappy run
if best_ams < 3.4:
	exit()

traindata = None # may help garbage collection free up some memory

write("loading test data")
testdata = pd.read_sql("SELECT * FROM test", conn)
writeDone()

write("creating custom features on testdata")
make_isnull_cols(testdata)
writeDone()

write("computing zscores on testdata")
for col in feature_cols:
	column = testdata[col]
	isna = np.isnan(column) | (column == -999.0)

	testdata.loc[isna, col] = means[col]
	testdata[col] = (testdata[col] - means[col]) / stddevs[col]
writeDone()

write("classifying test data")
classes, confidences = classifier.classify(testdata[feature_cols].values)
testdata["Class"] = classes
testdata["confidence"] = confidences
testdata = testdata.sort("confidence")
testdata["RankOrder"] = range(1, testdata.shape[0] + 1)
testdata = testdata.sort("EventId")
writeDone()

write("writing output to {0}".format(CSV_OUTPUT_FILE))
testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
zf.write(CSV_OUTPUT_FILE)
zf.close()
writeDone()
