import numpy as np
import pandas as pd
from bkputils import *

# import cProfile as profile

import bspkde, time, colflags, zipfile

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None
CSV_OUTPUT_FILE = "bspkde.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

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

def score_df(df):
	global feature_cols
	df["row_col_flags"] = colflags.calc_col_flags(df, feature_cols)
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

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT is not None else TRAIN_LIMIT)
traindata = traindata[:TRAIN_LIMIT]
feature_cols = featureCols(only_float64=True)
traindata["row_col_flags"] = colflags.calc_col_flags(traindata, feature_cols)
writeDone()
print "	num rows:", traindata.shape[0]
print

comparator_set_lookup = {}
for col_flags, group in traindata.groupby("row_col_flags"):
	# if col_flags != 2**29-1:
	# 	continue
	col_flag_str = "{0:b}".format(col_flags)
	write("building BspKdeComparatorSet for %s" % col_flag_str)
	comparator_set_lookup[col_flags] = comparator = bspkde.BspKdeComparatorSet(
		col_flag_str,
		group,
		colflags.get_flagged_cols(col_flags, feature_cols)
	)
	writeDone()
	print "	max depth: %d" % comparator.get_max_depth()

write("classifying training rows")
# profile.run("traindata = score_df(traindata)", "bspkde.profile")
traindata = score_df(traindata)
writeDone()
print "AMS:", ams(traindata["Class"], traindata)
print "signal (actual): %2.2f%%" % (float(np.sum(traindata["Label"]=="s")) * 100.0 / np.sum(traindata.shape[0]))
print "signal (predicted): %2.2f%%" % (float(np.sum(traindata["Class"]=="s")) * 100.0 / np.sum(traindata.shape[0]))

write("loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

write("classifying test data")
# profile.run("testdata = score_df(testdata)")
testdata = score_df(testdata)
writeDone()

write("writing output")
testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
zf.write(CSV_OUTPUT_FILE)
zf.close()
writeDone()

writeDone(time.time() - global_start)