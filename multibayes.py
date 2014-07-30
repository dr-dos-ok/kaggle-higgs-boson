import numpy as np
import pandas as pd
from bkputils import *
from voronoi import *
import math, time, sys, zipfile, json, random, scipy.spatial

global_start = time.time()

random.seed(42)

TRAIN_LIMIT = None
TEST_LIMIT = None
CSV_OUTPUT_FILE = "multibayes.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

def calc_col_flags(df):
	row_sums = np.zeros(df.shape[0], dtype=np.int)
	for index, col in enumerate(feature_cols):
		series = df[col]
		row_sums += np.logical_not(np.isnan(series)) * (2**index)
	return row_sums

def get_flagged_cols(col_flags, available_cols = None):
	global feature_cols
	if available_cols == None:
		available_cols = feature_cols
	return [
		col for (index, col) in enumerate(available_cols)
		if col_flags & 2**index > 0
	]

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

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT is not None else None)
traindata = traindata[:TRAIN_LIMIT]
feature_cols = featureCols(only_float64=True)
writeDone()
print "	num rows:", traindata.shape[0]
print

comparator_set_lookup = {}
traindata["row_col_flags"] = calc_col_flags(traindata)
for col_flags, group in traindata.groupby("row_col_flags"):
	write("building VoronoiKdeComparatorSet for {0:b}".format(col_flags))
	comparator_set_lookup[col_flags] = comparator = VoronoiKdeComparatorSet(
		"{0:b}".format(col_flags),
		group,
		get_flagged_cols(col_flags)
	)
	writeDone()

print
print_timers()
print

def score_df(df):
	df["row_col_flags"] = calc_col_flags(df)
	df["Class"] = ["MONKEY"] * df.shape[0]
	df["confidence"] = ["MONKEY"] * df.shape[0]
	for col_flags, group in df.groupby("row_col_flags"):
		if col_flags in comparator_set_lookup:
			comparator_set = comparator_set_lookup[col_flags]
			class_, confidence = comparator_set.classify(group)
			df["Class"][group.index] = class_
			df["confidence"][group.index] = confidence
	df = df.sort("confidence")
	df["RankOrder"] = range(1, df.shape[0] + 1)
	df = df.sort("EventId")
	return df

write("classifying training rows")
traindata = score_df(traindata)
writeDone()

print "AMS:", ams(traindata["Class"], traindata)
print "signal (actual): %2.2f%%" % (float(np.sum(traindata["Label"]=="s")) * 100.0 / np.sum(traindata.shape[0]))
print "signal (predicted): %2.2f%%" % (float(np.sum(traindata["Class"]=="s")) * 100.0 / np.sum(traindata.shape[0]))

write("loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

write("classifying test data")
testdata = score_df(testdata)
writeDone()

write("writing output")
testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
zf.write(CSV_OUTPUT_FILE)
zf.close()
writeDone()

writeDone(time.time() - global_start)