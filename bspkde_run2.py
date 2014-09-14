import numpy as np
import numpy.ma as ma
import pandas as pd
from bkputils import *

import bspkde, time, colflags, zipfile, scipy.stats, multiprocessing

NUM_MODELS = 100
COLS_PER_MODEL = 6
TRAIN_LIMIT = None
TEST_LIMIT = None
CSV_OUTPUT_FILE = "bspkde2.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

print "NUM_MODELS:", NUM_MODELS
print "COLS_PER_MODEL:", COLS_PER_MODEL
print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

_parallel_score_dataframe = None
def parallel_score(comparator):
	global _parallel_score_dataframe
	return comparator.score(_parallel_score_dataframe)

def serial_score(comparator, dataframe):
	return comparator.score(dataframe)

class RandomComparatorSet(object):
	def __init__(self, column_sets, feature_cols, dataframe):

		num_models = len(column_sets)
		self.comparators = [None] * num_models
		for i in xrange(num_models):
			model_cols = column_sets[i]
			self.comparators[i] = bspkde.BspKdeComparator(",".join(model_cols), dataframe, model_cols)

	def score(self, dataframe, parallel):
		global _parallel_score_dataframe
		n_comparators = len(self.comparators)

		_parallel_score_dataframe = dataframe

		if parallel==True:
			_parallel_score_dataframe = dataframe
			pool = multiprocessing.Pool()
		elif parallel==False:
			pool=None
		else:
			pool=parallel #assumed to be instance of multiprocessing.Pool()

		if pool is not None:
			scores = pool.map(parallel_score, self.comparators)
		else:
			scores = map(serial_score, self.comparators, [dataframe] * len(self.comparators))

		s_scores, b_scores = zip(*scores)

		s_scores = ma.masked_invalid(s_scores)
		b_scores = ma.masked_invalid(b_scores)

		return (
			scipy.stats.mstats.gmean(s_scores, axis=0).data,
			scipy.stats.mstats.gmean(b_scores, axis=0).data
		)

class Classifier(object):
	def __init__(self, comparator):
		self.comparator = comparator
		self.cutoff = 0.0
		self.options = np.array(["b", "s"])

	def classify(self, dataframe, parallel=None):
		score_s, score_b = self.comparator.score(dataframe, parallel=parallel)

		diffs = score_s - score_b
		ratios = score_s / score_b
		classes = self.options[(ratios > self.cutoff).astype(np.int)]
		confidences = (diffs - self.cutoff) ** 2

		return (classes, confidences)

write("Running random bspkde model averages")

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT is not None else TRAIN_LIMIT)
traindata = traindata[:TRAIN_LIMIT]
feature_cols = featureCols(only_float64=True)
writeDone()

# write("creating all 2D column sets")
# column_sets = [a for a in bspkde.choose(feature_cols, 2)][:NUM_MODELS]
# writeDone()

write("creating {0:d} random {1:d}D column sets".format(NUM_MODELS, COLS_PER_MODEL))
column_sets = [np.random.choice(feature_cols, COLS_PER_MODEL, replace=False) for i in xrange(NUM_MODELS)]
writeDone()

write("creating ComparatorSet")
comparator = RandomComparatorSet(column_sets, feature_cols, traindata)
writeDone()

write("creating and tuning classifier")
def test_cutoff(exponent):
	cutoff = np.exp(exponent)
	classifier = Classifier(comparator)
	classifier.cutoff = cutoff
	predictions, confidence = classifier.classify(traindata, parallel=False)
	score = ams(predictions, traindata)
	return (exponent, cutoff, score)
cutoffs = np.arange(-0.5, 0.6, 0.1)
results = multiprocessing.Pool().map(test_cutoff, cutoffs)
best_cutoff = 0.0
best_ams = 0.0
best_exponent = np.nan
for exponent, cutoff, score in results:
	if score > best_ams:
		best_ams = score
		best_cutoff = cutoff
		best_exponent = exponent
classifier = Classifier(comparator)
classifier.cutoff = best_cutoff
writeDone()
print "\t\tcutoffs:", cutoffs
print "\t\tcutoff: {0:f} (e^{1:f})".format(classifier.cutoff, best_exponent)
print "\t\tpredicted ams: {0:f}".format(best_ams)
# for cutoff, score in zip(cutoffs, all_ams):
# 	print "\t\t\t{0:f}: {1:f}".format(cutoff, score)

if best_ams < 3.4:
	writeDone()
	exit()

traindata = None # may help with memory

write("loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

write("classifying test data")
predictions, confidence = classifier.classify(testdata, parallel=True)
testdata["Class"] = predictions
testdata["confidence"] = confidence
testdata = testdata.sort("confidence")
testdata["RankOrder"] = range(1, testdata.shape[0]+1)
testdata = testdata.sort("EventId")
writeDone()

write("writing output to {0}".format(CSV_OUTPUT_FILE))
testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
zf.write(CSV_OUTPUT_FILE)
zf.close()
writeDone()

writeDone()