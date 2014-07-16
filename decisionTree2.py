import numpy as np
import pandas as pd
from bkputils import *
import math, time

global_start = time.time()

TRAIN_LIMIT = 10000

class BinSet_Real:
	def __init__(self, dataframe, col):

		self.col = col

		npData = dataframe[col].dropna().values
		self.probna = probna = 1.0 - float(len(npData))/dataframe.shape[0]

		#custom bin slices such that each bin contains an equal number
		#of observations. Gives us more granularity at the more probable
		#points of the distribution so that we don't have dozens of bins
		#with just a few data points in them
		sortedData = np.sort(npData)
		self.numbins = numBins = int(math.sqrt(npData.size))
		bindices = np.linspace(0, npData.size-1, numBins+1).astype(np.int32) # bin + indices = bindices! lol I'm hilarious...
		bins = sortedData[bindices]

		#generate histogram density
		(self._hist, self._bins) = np.histogram(npData, bins, density=True)
		self._maxval = self._bins[-1]
		self._maxbindex = numBins # don't subtract one - numpy thinks bins are 1-based not 0-based

		#account for probability of NA values
		# probna + sum(bin_density*bin_width) needs to be equal to 1.0
		self._hist = self._hist / (1.0 - probna)

		#add NaN to either end of our hist array for two reasons:
		# 1 - numpy.digitize will now return the correct index of the bin. Don't have to worry about
		#		index-out-of-bounds issues
		# 2 - test data outside of our observed range will now produce numpy.nan for a probability density (instead
		#		of density=0, which is clearly false - we just don't know what it is, therefore it's
		#		better to not make assumptions and just ignore this outlier)
		self._hist = np.array([np.nan] + (list(self._hist) + [np.nan]))

	def _getbindices(self, datapoints):
		if isinstance(datapoints, pd.Series):
			datapoints = datapoints.values

		#outsource the heavy lifting...
		result = np.digitize(datapoints, self._bins)

		#special value to indicate nan indices (can't have int nan or I would use that)
		#I prefer this to np.digitize's normal behavior of returning "above highest bound"
		#index.
		#NOTE: Although -1 is not a valid bin, it is a valid array index, so using the results of this
		#method to index into an ndarray will silently fail and give you wrong results. Code calling this
		#method should be aware of this and account for it manually
		result[np.isnan(datapoints)] = -1

		#account for a minor inconsistency in np.digitize: final bin should
		#include its upper limit instead of being treated as out-of-bounds
		result[datapoints == self._maxval] = self._maxbindex

		return result

	def score(self, data):
		bindices = self._getbindices(data)
		notna = bindices != -1

		result = np.empty(bindices.shape)
		result[notna] = self._hist[bindices[notna]]
		result[not notna] = np.na

		return result

	def entropy(self, dataframe):
		"""
		Returns the entropy of the given dataframe with respect to this
		BinSet's chosen column and calculated bins. The column is assumed to
		be real-valued so binning is necessary before calculating an
		entropy value
		"""

		bindices = pd.Series(self._getbindices(dataframe[self.col]))

		colValCounts = bindices.groupby(lambda index: bindices[index]).size()
		_sum = float(np.sum(colValCounts))
		if _sum	== 0.0:
			return 0.0
		colValProbs	= colValCounts / _sum
		colValProbs = colValProbs[colValProbs != 0.0]
		H = -np.sum(colValProbs * np.log2(colValProbs))
		return H

class BinSet_Discrete:
	def __init__(self, dataframe, col):
		self.col = col

		series = dataframe[col]
		groups = series.groupby(lambda index: series[index])

		self.bins = bins = {}
		for key, group in groups:
			bins[key] = float(group.size) / series.size
		self.numbins = len(self.bins)

	def score(self, data):
		return np.array([
			self.bins[key] if key in self.bins else np.nan
			for key in data
		])

	def entropy(self, dataframe):
		colValCounts = dataframe.groupby([self.col]).size()
		_sum = float(np.sum(colValCounts))
		if _sum == 0.0:
			return 0.0
		colValProbs = colValCounts / _sum
		colValProbs = colValProbs[colValProbs != 0.0] # 0*log(0)==0 but log(0) gives NaN so remove these
		H = -np.sum(colValProbs * np.log2(colValProbs))
		return H


class BinSet:
	def __init__(self, dataframe, col):
		dtype = dataframe[col].dtype
		if dtype == np.float64:
			self.delegate = BinSet_Real(dataframe, col)
		else:
			self.delegate = BinSet_Discrete(dataframe, col)
		self.numbins = self.delegate.numbins

	def score(self, data):
		return self.delegate.score(data)

	def entropy(self, dataframe):
		return self.delegate.entropy(dataframe)

	def mutualinfo(self, dataframe, othercol):
		marginal_entropy = self.delegate.entropy(dataframe)

		sizes = dataframe.groupby(othercol).size()
		total = float(np.sum(sizes))
		sum_conditional = 0.0

		for val in pd.unique(dataframe[othercol]):
			val_prob = sizes[val] / total
			subdf = dataframe[dataframe[othercol] == val]
			val_entropy = self.delegate.entropy(subdf)
			sum_conditional += (val_prob * val_entropy)

		return marginal_entropy - sum_conditional

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
featureCols = featureCols(only_float64=False)
writeDone()

def printbin(col):
	binset = BinSet(traindata, col)
	print "%s\t[%d]\t%f\t%f" % (
		# index,
		col.ljust(30, " "),
		binset.numbins,
		binset.entropy(traindata),
		binset.mutualinfo(traindata, "Label")
	)

for index, col in enumerate(featureCols):
	printbin(col)
# printbin("PRI_jet_num")

writeDone(time.time() - global_start)