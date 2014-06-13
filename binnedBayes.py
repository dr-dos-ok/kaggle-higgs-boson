import numpy as np
import pandas as pd
from bkputils import *
import math, time, sys

global_start = time.time()

class BinSet:
	def __init__(self, dataframe):

		npData = dataframe.ix[:,0]
		npWeights = dataframe.ix[:,1]

		#custom bin slices such that each bin contains an equal number
		#of observations. Gives us more granularity at the more probable
		#points of the distribution so that we don't have dozens of bins
		#with just a few data points in them
		sortedData = np.sort(npData)
		self._numBins = numBins = int(math.sqrt(npData.size))
		bindices = np.linspace(0, npData.size-1, numBins+1).astype(np.int32) # bin + indices = bindices! lol I'm hilarious...
		bins = sortedData[bindices]

		#generate histogram density
		(self._hist, self._bins) = np.histogram(npData, bins, weights=npWeights, density=True)

		#add NaN to either end of our hist array for two reasons:
		# 1 - numpy.digitize will now return the correct index of the bin. Don't have to worry about
		#		index-out-of-bounds issues
		# 2 - test data outside of our observed range will now produce numpy.nan for a probability density (instead
		#		of density=0, which is clearly false - we just don't know what it is, therefore it's
		#		better to not make assumptions and just ignore this outlier)
		self._hist = np.array([np.nan] + list(self._hist) + [np.nan])

	def score(self, data):
		return self._hist[np.digitize(data, self._bins)]

conn = dbConn()

write("loading training data")
traindata = pd.read_sql("SELECT * FROM training", conn)
traindata = traindata.applymap(lambda x: np.nan if x == -999.0 else x)
colNames = list(traindata.columns.values)
featureCols = [colName for colName in colNames if traindata.dtypes[colName] == np.float64]
featureCols.remove("Weight")
traindata_b = traindata[traindata["Label"] == "b"]
traindata_s = traindata[traindata["Label"] == "s"]
traindata = None # don't need this anymore, might clean up global namespace a bit
writeDone()

write("binning training data")
binsets_b = dict(zip(
	featureCols,
	(BinSet(traindata_b[[col, "Weight"]].dropna(how="any")) for col in featureCols)
))
binsets_s = dict(zip(
	featureCols,
	(BinSet(traindata_s[[col, "Weight"]].dropna(how="any")) for col in featureCols)
))
traindata_s = None
traindata_b = None
writeDone()

write("loading test data")
sql = "SELECT EventId, %s FROM test" % (", ".join(featureCols))
testdata = pd.read_sql(sql, conn)
testdata = testdata.applymap(lambda x: np.nan if x == -999.0 else x)
writeDone()

write("generating scores")
scores_b = pd.DataFrame()
scores_s = pd.DataFrame()
for colName in featureCols:
	scores_b[colName] = binsets_b[colName].score(testdata[colName])
	scores_s[colName] = binsets_s[colName].score(testdata[colName])
writeDone()

write("consolidating scores and writing output")
posterior_b = scores_b.prod(axis=1)
posterior_s = scores_s.prod(axis=1)
testdata["posterior_diff"] = posterior_s - posterior_b
testdata["Class"] = ["s" if posterior_diff > 0. else "b" for posterior_diff in testdata["posterior_diff"]]

testdata = testdata.sort("posterior_diff")
testdata["RankOrder"] = range(1, testdata.shape[0]+1)
testdata = testdata.sort("EventId")

testdata[["EventId", "RankOrder", "Class"]].to_csv("binnedBayes.csv", header=True, index=False)
writeDone()

global_elapsed = time.time() - global_start
print "Took %d:%d" % (global_elapsed/60, global_elapsed%60)






























