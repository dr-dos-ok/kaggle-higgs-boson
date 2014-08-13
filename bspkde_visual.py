import numpy as np
import pandas as pd
from bkputils import *

import bspkde, colflags, time, random

import matplotlib.pyplot as pyplot
import matplotlib as mpl
from matplotlib.collections import PolyCollection

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None

seed = 42
random.seed(seed)
np.random.seed(seed)

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT is not None else TRAIN_LIMIT)
traindata = traindata[:TRAIN_LIMIT]
feature_cols = featureCols(only_float64=True)
# feature_cols = ["DER_met_phi_centrality", "DER_lep_eta_centrality"]
traindata["row_col_flags"] = colflags.calc_col_flags(traindata, feature_cols)
# traindata = traindata[traindata["row_col_flags"] == 3] #2**29-1] # TODO - remove this
writeDone()
print "	num rows:", traindata.shape[0]
print

comparator_set_lookup = {}
for col_flags, group in traindata.groupby("row_col_flags"):
	if col_flags != 2**29-1:
		continue
	col_flag_str = "{0:b}".format(col_flags)
	write("building VoronoiKdeComparatorSet for %s" % col_flag_str)
	comparator_set_lookup[col_flags] = comparator = bspkde.BspKdeComparatorSet(
		col_flag_str,
		group,
		colflags.get_flagged_cols(col_flags, feature_cols)
	)
	writeDone()

pyplot.ion()
comparator_set_lookup[2**29-1].plot()