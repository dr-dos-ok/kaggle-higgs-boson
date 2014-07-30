import numpy as np
import pandas as pd
from bkputils import *
from voronoi import *

import matplotlib.pyplot as pyplot
import matplotlib as mpl
from matplotlib.collections import PolyCollection

import random

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None

seed = 49
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
feature_cols = featureCols(only_float64=True)
writeDone()
print "	num rows:", traindata.shape[0]
print

comparator_set_lookup = {}
traindata["row_col_flags"] = calc_col_flags(traindata)
for col_flags, group in traindata.groupby("row_col_flags"):
	if col_flags != 2**29 - 1:
		continue
	write("building VoronoiKdeComparatorSet for {0:b}".format(col_flags))
	comparator_set_lookup[col_flags] = comparator = VoronoiKdeComparatorSet(
		"{0:b}".format(col_flags),
		group,
		get_flagged_cols(col_flags)
	)
	writeDone()

pyplot.ion()
pyplot.show()
# for key in comparator_set_lookup:
# 	comparator_set_lookup[key].plot()
comparator_set_lookup[2**29 - 1].plot()

# kde = VoronoiKde(
# 	"test_kde",
# 	pd.DataFrame([
# 		#bin points
# 		[0.0, 0.0],		# 0
# 		[0.0, 4.0],		# 1
# 		[6.0, 0.0],		# 2
# 		[0.0, -8.0],	# 3
# 		[-2.0, 0.0],	# 4

# 		#fill points
# 		[-2.0, -1.0],	# 4
# 		[0.0, 3.0],		# 1
# 		[1.0, 0.0],		# 0
# 		[2.0, 0.0],		# 0
# 		[2.0, -2.0],	# 0
# 		[4.0, 0.0],		# 2
# 		[5.0, 0.0]		# 2
# 	]),
# 	[0, 1],
# 	bin_indices=range(5)
# )

# pyplot.ion()
# fig, ax = pyplot.subplots(1, 1)

# all_min = min(kde.bin_densities)
# all_max = max(kde.bin_densities)
# norm = mpl.colors.Normalize(vmin=all_min, vmax=all_max)

# kde.plot_heatmap(fig, ax, norm)

# ax.set_xlim([-3, 7])
# ax.set_ylim([-9, 5])

# pyplot.show()
# __ = raw_input("Enter to continue...")