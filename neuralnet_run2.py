import pandas as pd
import numpy as np
import neuralnet as nn
import sqlite3 as sqlite

from bkputils import *
from neuralnet_train import train
from neuralnet_classifier import ZNetClassifier

import os, zipfile, split

NET_SAVE_FILE = "neuralnet_run2.nn"
CSV_OUTPUT_FILE = "neuralnet2.csv"
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

write("building full-dataset classifier")

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

write("splitting data into validation and training sets")
validation_ids, training_ids = split.load_ids()
validation_set = traindata.loc[validation_ids]
training_set = traindata.loc[training_ids]
writeDone()

layer_sizes = [-1, 100, 100, 100, -1]
loaded_net = False
# if os.path.exists(NET_SAVE_FILE):
if False:
	write("loading saved net")
	net = nn.FeedForwardNet.load(NET_SAVE_FILE)
	writeDone()
	min_err = None
	if np.array_equal(net.layer_sizes, layer_sizes):
		loaded_net = True
	else:
		print "\t!!! parameters have changed; discarding net !!!"
	
if not loaded_net:
	write("creating and training new net")
	net = nn.ZFeedForwardNet(
		traindata,
		feature_cols, output_cols,
		layer_sizes,
		hidden_fn_pair=nn.TANH_MOD_FN_PAIR,
		output_layer=nn.SOFTMAX_CROSS_ENTROPY_OUTPUT_LAYER
	)
	min_err = train(net,
		validation_set,
		training_set,
		learning_rate=0.001,
		velocity_decay=0.99,
		batch_size=100
	)
	writeDone()
print "\t\tmin {0}: {1}".format(net.err_fn.__name__, (str(min_err) if min_err is not None else "(unknown)"))

write("saving net")
net.save(NET_SAVE_FILE)
writeDone()

write("searching for best cutoff value")
classifier = ZNetClassifier(net)
best_cutoff = 0.0
best_ams = 0.0
for cutoff in np.arange(0.25, 0.75, 0.01):
	classifier.cutoff = cutoff
	predictions, confidence = classifier.classify(validation_set)
	score = ams(predictions, validation_set)
	if score > best_ams:
		best_ams = score
		best_cutoff = cutoff
classifier.cutoff = best_cutoff
writeDone()
print "\t\tcutoff: {0}".format(classifier.cutoff)
print "\t\tprdicted ams: {0}".format(best_ams)

#don't bother classifying the test data if this was a crappy run
if best_ams >= 3.4:

	traindata = None # may help garbage collection free up some memory

	write("loading test data")
	testdata = pd.read_sql("SELECT * FROM test", conn)
	writeDone()

	write("creating custom features on testdata")
	make_isnull_cols(testdata)
	writeDone()

	write("classifying test data")
	# profile.run("testdata = score_df(testdata)")
	classes = np.empty(testdata.shape[0], dtype=np.dtype("S1"))
	confidences = np.empty(testdata.shape[0], dtype=np.float64)
	chunks = np.array_split(testdata, 5)
	start = 0
	for chunk in chunks:
		stop = start + chunk.shape[0]
		class_, confidence = classifier.classify(chunk)
		# print class_
		# print confidence
		# exit()
		classes[start:stop] = class_
		confidences[start:stop] = confidence
		start = stop
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

	writeDone()