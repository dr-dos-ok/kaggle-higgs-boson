import numpy as np
import pandas as pd
from bkputils import *
import math, time

TRAIN_LIMIT = None

def row_int(row):
	result = 0
	for index, col in enumerate(featureCols):
		if not np.isnan(row[col]):
			result += 2**index
	return result

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
featureCols = featureCols(only_float64=False)
writeDone()

write("calculating row ints")
row_ints = pd.Series([row_int(row) for index, row in traindata.iterrows()])
# for rint in row_ints:
# 	print "{0:b}".format(rint)
writeDone()

write("grouping rows")
groups = row_ints.groupby(lambda index: row_ints[index])
writeDone()

print
for key, group in groups:
	print "{0:b}".format(key), group.size