import pandas as pd
import numpy as np
import neuralnet as nn
import sqlite3 as sqlite

from bkputils import *
from neuralnet_train import train
from neuralnet_classifier import ZNetClassifier

import os, zipfile

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

def make_trainfeature_col(dataframe):
	dataframe["train_feature"] = np.ones(dataframe.shape[0])
	dataframe.loc[dataframe["Label"]=="b", "train_feature"] = -1

write("building full-dataset classifier")

write("loading training data")
conn = sqlite.connect("data.sqlite")
traindata = pd.read_sql("SELECT * FROM training", conn)
feature_cols = [col for col in traindata.columns.values if col.startswith("DER") or col.startswith("PRI")]
writeDone()

write("creating custom features on traindata")
make_isnull_cols(traindata)
make_trainfeature_col(traindata)
writeDone()

if os.path.exists(NET_SAVE_FILE):
	write("loading and re-training existing net")
	net = nn.FeedForwardNet.load(NET_SAVE_FILE)
	min_err = train(net, traindata, learning_rate=0.0001, velocity_decay=0.2)
	writeDone()
else:
	write("training new net")
	net = nn.ZFeedForwardNet(traindata, feature_cols, [-1, 1000, 100, 1])
	min_err = train(net, traindata, learning_rate=0.01, velocity_decay=0.99)
	writeDone()

write("saving net")
net.save(NET_SAVE_FILE)
writeDone()

write("classifying traindata")
classifier = ZNetClassifier(net)
predictions, confidence = classifier.classify(traindata)
score = ams(predictions, traindata)
writeDone()
print "\t\tpredicted ams: {0}".format(score)
print "\t\tmin squared err: " + (str(min_err) if min_err is not None else "(unknown)")
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
confidences = np.empty((testdata.shape[0], 1), dtype=np.float64)
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