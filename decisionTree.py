import numpy as np
import pandas as pd
from bkputils import *
import math, time

import matplotlib.pyplot as pyplot
import matplotlib as mpl

global_start = time.time()

def entropy(df, colName):
	colValCounts = df.groupby([colName]).size()
	_sum = float(np.sum(colValCounts))
	if _sum == 0.0:
		return 0.0
	colValProbs = colValCounts / _sum
	colValProbs = colValProbs[colValProbs != 0.0] # 0*log(0)==0 but log(0) gives NaN so remove these
	H = -np.sum(colValProbs * np.log2(colValProbs))
	return H

def mutualinfo(df, col1Name, col2Name):
	"""
	implementation using definition I[C;X] = H[C] - SUM(Pr(X=x)*H[C|X=x]) instead of
	weighted average. This is more efficient because it avoids recalculating the
	marginal entropy of df over and over (only does it once)
	"""
	marginalEntropy = entropy(df, col1Name)

	sizes = df.groupby(col2Name).size()
	total = float(np.sum(sizes))
	sum_conditional = 0.0

	for val in pd.unique(df[col2Name]):
		valProb = sizes[val] / total
		subdf = df[df[col2Name] == val]
		valEntropy = entropy(subdf, col1Name)
		sum_conditional += (valProb * valEntropy)

	return marginalEntropy - sum_conditional

class BinSet:
	def __init__(self, dataframe, col):
		self.col = col

		npData = dataframe.ix[:,0]

		#custom bin slices such that each bin contains an equal number
		#of observations. Gives us more granularity at the more probable
		#points of the distribution so that we don't have dozens of bins
		#with just a few data points in them
		sortedData = np.sort(npData)
		self._numBins = numBins = int(math.sqrt(npData.size))
		bindices = np.linspace(0, npData.size-1, numBins+1).astype(np.int32) # bin + indices = bindices! lol I'm hilarious...
		bins = sortedData[bindices]

		#generate histogram density
		(self._hist, self._bins) = np.histogram(npData, bins, density=True)

		#add NaN to either end of our hist array for two reasons:
		# 1 - numpy.digitize will now return the correct index of the bin. Don't have to worry about
		#		index-out-of-bounds issues
		# 2 - test data outside of our observed range will now produce numpy.nan for a probability density (instead
		#		of density=0, which is clearly false - we just don't know what it is, therefore it's
		#		better to not make assumptions and just ignore this outlier)
		self._hist = np.array([np.nan] + (list(self._hist) + [np.nan]))

	@staticmethod
	def fromdataframe(dataframe, col):
		return BinSet(dataframe[[col]], col)

	def getbindex(self, row):
		return np.digitize(row[self.col], self._bins)

	def getbindices(self, datapoints):
		retun np.digitize(datapoints, self._bins)

	def binnedEntropy(self, dataframe):
		"""
		Returns the entropy of the given dataframe with respect to this
		BinSet's chosen column and calculated bins. The column is assumed to
		be real-valued so binning is necessary before calculating an
		entropy value
		"""

		bindices = self.getbindices(dataframe[self.col])

		colValCounts = df.groupby([self.col]).size()
		_sum = float(np.sum(colValCounts))
		if _sum	== 0.0:
			return 0.0
		colValProbs	= colValCounts / _sum
		colValProbs = colValProbs[colValProbs != 0.0]
		H = -np.sum(colValProbs * np.log2(colValProbs))
		return H

	def mutualinfo(self, dataframe, othercol):
		"""
		Calculate the mutual information of this BinSet's chosen column
		with the passed othercol, with respect to the passed dataframe.
		The othercol is assumed to be discretely-valued (not real-valued)
		and so no binning is necessary (this is me being lazy because I
		know that will always be true with my current problem).
		"""

		marginalEntropy = self.binnedEntropy(dataframe)

		sizes = dataframe.groupby(othercol).size()
		total = float(np.sum(sizes))
		sum_conditional = 0.0

		for val in pd.unique(dataframe[othercol]):
			valProb = sizes[val] / total
			subdf = dataframe[dataframe[othercol] == val]
			valEntropy = entropy(subdf, othercol)
			sum_conditional += (valProb * valEntropy)

		return marginalEntropy - sum_conditional

class DecisionTree:
	def __init__(self, dataframe, remaining_cols, labelcol):
		self.dataframe = dataframe
		self.remaining_cols = remaining_cols
		self.labelcol = labelcol
		self.haschildren = False

	def grow(self):
		if not self.haschildren:
			self.makechildren()
		else:
			maxentropy = 0.0
			maxindex = -1
			for index, child in enumerate(self.children):
				childentropy = child.getentropy()
				if childentropy > maxentropy:
					maxentropy = childentropy
					maxindex = index
			self.children[maxindex].grow()

	def makechildren(self):
		binsets = [
			BinSet(self.dataframe, col)
			for col in self.remaining_cols
		]
		mutualinfos = np.array([
			binset.mutualinfo(self.dataframe, self.labelcol))
			for binset in BinSet
		])
		maxindex = mutualinfos.argmax()
		maxcol = self.remaining_cols[maxindex]
		maxbinset = binsets[maxindex]
		self.childbins = maxbinset

		

	def decide(self, row):
		bindex = self.binset.getbindex(row)
		self.children[bindex].decide(row)

class DecisionLeaf:
	def __init__(self, decision, remaining_entropy):
		self.decision = decision
		self.remaining_entropy = remaining_entropy

	def decide(self, row):
		return self.decision

write("loading training data")
traindata = loadTrainingData(trainLimit)
featureCols = featureCols()
writeDone()

root = DecisionTree(traindata, featureCols, "Label")
for i in range(10):
	root.grow()

writeDone(time.time() - global_start)