import numpy as np
import pandas as pd
from bkputils import *

import bspkde, time, random

import matplotlib.pyplot as pyplot
import matplotlib as mpl
from matplotlib.collections import PolyCollection

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None

seed = 42
random.seed(seed)
np.random.seed(seed)

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

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT is not None else TRAIN_LIMIT)
traindata = traindata[:TRAIN_LIMIT]
# feature_cols = featureCols(only_float64=True)
feature_cols = ["DER_mass_MMC", "DER_mass_transverse_met_lep"]
traindata["row_col_flags"] = calc_col_flags(traindata)
traindata = traindata[traindata["row_col_flags"] == 3] #2**29-1] # TODO - remove this
writeDone()
print "	num rows:", traindata.shape[0]
print

min_corner = np.amin(traindata[feature_cols].values, axis=0)
max_corner = np.amax(traindata[feature_cols].values, axis=0)
diff = max_corner - min_corner
margin = 0.05 * diff
max_corner = max_corner + margin
min_corner = min_corner - margin

p = bspkde.Partition(traindata[feature_cols].values, min_corner, max_corner)
p.train()
print "num leaves:", p.count_leaf_children()

pyplot.ion()
p.plot()
__ = raw_input("Enter to exit...")