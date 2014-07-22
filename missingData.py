import numpy as np
import pandas as pd
from bkputils import *
import math, time

TRAIN_LIMIT = None

# def row_int(row):
# 	result = 0
# 	for index, col in enumerate(featureCols):
# 		if not np.isnan(row[col]):
# 			result += 2**index
# 	return result

def calc_row_ints(df):
	row_sums = pd.Series(np.zeros(df.shape[0], dtype=np.int32))
	for index, col in enumerate(featureCols):
		series = df[col]
		row_sums += np.logical_not(np.isnan(series)) * (2**index)
	return row_sums

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
featureCols = featureCols(only_float64=False)
writeDone()

write("calculating row ints")
# row_ints = pd.Series([row_int(row) for index, row in traindata.iterrows()])
row_ints = calc_row_ints(traindata)
# for rint in row_ints:
# 	print "{0:b}".format(rint)
writeDone()

write("grouping rows")
groups = row_ints.groupby(lambda index: row_ints[index])
writeDone()

featureCols.reverse()
for col in featureCols:
	print col

print
for key, group in groups:
	print "{0:b}".format(key), group.size