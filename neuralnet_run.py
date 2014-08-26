import numpy as np
import pandas as pd
from bkputils import *
import neuralnet as nn

from zwrapper import ZWrapper
from colswitch import ColSwitchClassifier

import time, zipfile, random, sys

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None
CSV_OUTPUT_FILE = "neuralnet.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

BATCH_SIZE = 100
LEARNING_RATE = 0.001
VELOCITY_DECAY = 0.99
ITERATIONS = 100

HIDDEN_LAYER_SIZES = [1000]

def ams(predictions, actual_df):
	predicted_signal = predictions == "s"
	actual_signal = actual_df["Label"] == "s"
	actual_background = ~actual_signal

	s = true_positives = np.sum(actual_df[predicted_signal & actual_signal]["Weight"])
	b = false_positives = np.sum(actual_df[predicted_signal & actual_background]["Weight"])
	B_R = 10.0

	radicand = 2.0 * ((s+b+B_R) * math.log(1.0 + (s/(b+B_R))) - s)

	if radicand < 0.0:
		print "radicand is less than 0, exiting"
		exit()
	else:
		return math.sqrt(radicand)

def build_znet_classifier(col_flag, available_cols, dataframe):
	layer_sizes = [len(available_cols)] + HIDDEN_LAYER_SIZES + [1]
	net = nn.ZFeedForwardNet(dataframe, available_cols, layer_sizes)

	write("training net for {0:b}".format(col_flag))
	train(net, col_flag, available_cols, dataframe)
	writeDone()
	return net

def train(net, col_flag, available_cols, dataframe):
	global BATCH_SIZE, LEARNING_RATE, VELOCITY_DECAY, ITERATIONS

	weights = net.get_flattened_weights()
	velocity = net.zeros_like_flattened_weights()
	index = 0
	print
	for iteration in xrange(ITERATIONS):
		batch_indices = random.sample(dataframe.index, min(BATCH_SIZE, dataframe.shape[0]))
		batch = dataframe.ix[batch_indices]

		grad = net.get_partial_derivs(
			batch[available_cols],
			batch[["train_feature"]],
			outputs=nn.FLATTENED_WEIGHTS
		)

		velocity *= VELOCITY_DECAY
		velocity += -grad * LEARNING_RATE
		weights += velocity
		net.set_flattened_weights(weights)

		if index % 10 == 0:
			forward = net.forward(dataframe[available_cols])

			errs, err_derivs = nn.squared_error(
				forward,
				dataframe[["train_feature"]].values
			)
			avgerr = np.mean(errs)
			sys.stdout.write("err: %1.9f\r" % (avgerr))
			sys.stdout.flush()
	print

	forward = net.forward(dataframe[available_cols])
	predict_vals = np.array(["b", "s"])
	s_predictions = predict_vals[(forward > 0.0).ravel().astype(np.int8)]
	print "ams: %f" % (ams(s_predictions, dataframe))

# def score_df(df):
# 	global feature_cols
# 	df["row_col_flags"] = colflags.calc_col_flags(df, feature_cols)
# 	df["Class"] = ["MONKEY"] * df.shape[0]
# 	df["confidence"] = ["MONKEY"] * df.shape[0]
# 	for col_flags, group in df.groupby("row_col_flags"):
# 		if col_flags in comparator_set_lookup:
# 			comparator_set = comparator_set_lookup[col_flags]
# 			class_, confidence = comparator_set.classify(group)
# 			df["Class"][group.index] = class_
# 			df["confidence"][group.index] = confidence
# 	df = df.sort("confidence")
# 	df["RankOrder"] = range(1, df.shape[0] + 1)
# 	df = df.sort("EventId")
# 	return df

def score_df(df):
	raise Exception("Not Finished")
	# global feature_cols, col_flags_lookup, net_lookup

	# df["row_col_flags"] = colflags.calc_col_flags(df, feature_cols)
	# df["Class"] = ["-"] * df.shape[0]
	# df["confidence"] = np.empty(df.shape[0])

	# for col_flags, group in df.groupby("row_col_flags"):
	# 	if col_flags in col_flags_lookup:

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print
print "BATCH_SIZE:", BATCH_SIZE
print "LEARNING_RATE:", LEARNING_RATE
print "VELOCITY_DECAY:", VELOCITY_DECAY
print "ITERATIONS:", ITERATIONS
print "HIDDEN_LAYER_SIZES:", HIDDEN_LAYER_SIZES
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT is not None else TRAIN_LIMIT)
traindata = traindata[:TRAIN_LIMIT]
feature_cols = featureCols(only_float64=True)
writeDone()
print "	num rows:", traindata.shape[0]
print

write("massaging training data")
traindata["train_feature"] = np.ones(traindata.shape[0])
traindata.loc[traindata["Label"]=="b", "train_feature"] = -1
writeDone()

write("building classifier")
classifier = ColSwitchClassifier(traindata, feature_cols, build_znet_classifier)
writeDone()
