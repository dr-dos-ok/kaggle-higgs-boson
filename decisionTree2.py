import numpy as np
import pandas as pd
from bkputils import *
import math, time

global_start = time.time()

TRAIN_LIMIT = 10000
TEST_LIMIT = 100

class BinSet_Real:
	def __init__(self, dataframe, col):

		self.original_data = dataframe
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

		self.original_groups = self.group(dataframe)

	def group(self, dataframe):
		bindices = self._getbindices(dataframe[self.col])
		# return dataframe.groupby(lambda index: bindices[index])

		# def mylambda(index):
		# 	print index
		# 	try:
		# 		return bindices[index]
		# 	except:
		# 		print "index:", index
		# 		print "bindices[%d]:" % len(bindices), bindices
		# 		print "numrows:", dataframe.shape
		# 		exit()
		return dataframe.groupby(bindices)

	def groupkeys(self, datapoints):
		return self._getbindices(datapoints)

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

	def entropy(self, dataframe=None):
		"""
		Returns the entropy of the given dataframe with respect to this
		BinSet's chosen column and calculated bins. The column is assumed to
		be real-valued so binning is necessary before calculating an
		entropy value
		"""

		if dataframe is None:
			# dataframe = self.original_data
			groups = self.original_groups
		else:
			if dataframe.shape[0] == 0:
				return 0.0
			groups = self.group(dataframe)

		# bindices = pd.Series(self._getbindices(dataframe[self.col]))
		# colValCounts = bindices.groupby(lambda index: bindices[index]).size()
		colValCounts = groups.size()
		_sum = float(np.sum(colValCounts))
		if _sum	== 0.0:
			return 0.0
		colValProbs	= colValCounts / _sum
		colValProbs = colValProbs[colValProbs != 0.0]
		H = -np.sum(colValProbs * np.log2(colValProbs))
		return H

class BinSet_Discrete:
	def __init__(self, dataframe, col):
		self.original_data = dataframe
		self.col = col

		series = dataframe[col]
		self.original_groups = groups = series.groupby(lambda index: series[index])

		self.bins = bins = {}
		for key, group in groups:
			bins[key] = float(group.size) / series.size
		self.numbins = len(self.bins)

	def group(self, dataframe):
		return dataframe.groupby(self.col)

	def groupkeys(self, datapoints):
		#since datapoints is assumed to be values from a discretely-valued column,
		#they are their own group keys already (this is different for real-valued
		#columns where group key is a bindex)
		return datapoints

	def score(self, data):
		return np.array([
			self.bins[key] if key in self.bins else np.nan
			for key in data
		])

	def entropy(self, dataframe=None):
		if dataframe is None:
			# dataframe = self.original_data
			groups = self.original_groups
		else:
			try:
				groups = dataframe.groupby([self.col])
			except:
				print "col:", col
				print dataframe
				exit()

		colValCounts = groups.size()
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
			self.delegate = delegate = BinSet_Real(dataframe, col)
		else:
			self.delegate = delegate = BinSet_Discrete(dataframe, col)

		self.numbins = delegate.numbins
		self.original_groups = delegate.original_groups
		self.original_data = delegate.original_data
		self.col = delegate.col

	def group(self, dataframe):
		return self.delegate.group(dataframe)

	def groupkeys(self, datapoints):
		return self.delegate.groupkeys(datapoints)

	def score(self, data):
		return self.delegate.score(data)

	def entropy(self, dataframe=None):
		return self.delegate.entropy(dataframe)

	def mutualinfo(self, othercol, dataframe=None):
		if dataframe is None:
			dataframe = self.delegate.original_data

		marginal_entropy = self.delegate.entropy(dataframe)

		groups = dataframe.groupby(othercol)
		total = float(dataframe.shape[0])
		sum_conditional = 0.0
		for key, subdf in groups:
			group_prob = subdf.shape[0] / total
			group_entropy = self.delegate.entropy(subdf)
			sum_conditional += (group_prob * group_entropy)

		return marginal_entropy - sum_conditional

		# sizes = dataframe.groupby(othercol).size()
		# total = float(np.sum(sizes))
		# sum_conditional = 0.0

		# for val in pd.unique(dataframe[othercol]):
		# 	val_prob = sizes[val] / total
		# 	subdf = dataframe[dataframe[othercol] == val]
		# 	val_entropy = self.delegate.entropy(subdf)
		# 	sum_conditional += (val_prob * val_entropy)

		# return marginal_entropy - sum_conditional

class DecisionTree:
	def __init__(self, binset, decisioncol):
		self.binset = binset
		self.decisioncol = decisioncol
		self.col = self.binset.col

		self.children = {}
		for key, group in binset.original_groups:
			self.children[key] = DecisionLeaf(dataframe=group, decisioncol=decisioncol)

	def decide(self, rows):
		groupkeys = self.binset.groupkeys(rows[self.col])
		rowgroups = rows.groupby(groupkeys)

		decisions = [None] * rows.shape[0]
		for groupkey, rowgroup in rowgroups:
			groupdecisions = self.children[groupkey].decide(rowgroup)
			for i, rowindex in enumerate(rowgroup.index):
				decisions[rowindex] = groupdecisions[i]

		return decisions

class DecisionLeaf:
	def __init__(self, dataframe=None, decisioncol=None, decision=None):
		if (dataframe is not None) and (decisioncol is not None):
			groups = dataframe.groupby(decisioncol)
			maxkey = None
			maxsize = 0
			for key, data in groups:
				if data.shape[0] > maxsize:
					maxkey = key
					maxsize = data.shape[0]
			self.decision = maxkey
		elif decision is not None:
			self.decision = decision
		else:
			raise Exception("DecisionLeaf requires either (dataframe,decisioncol) or (decision)")

	def decide(self, rows):
		return [self.decision] * len(rows)

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
featureCols = featureCols(only_float64=False)
writeDone()

# def printbin(col):
# 	binset = BinSet(traindata, col)
# 	print "%s\t[%d]\t%f\t%f" % (
# 		# index,
# 		col.ljust(30, " "),
# 		binset.numbins,
# 		binset.entropy(),
# 		binset.mutualinfo("Label")
# 	)

# for index, col in enumerate(featureCols):
# 	printbin(col)
# printbin("PRI_jet_num")

def find_greatest_mutualinfo(dataframe, remaining_cols):
	binsets = [BinSet(dataframe, col) for col in remaining_cols]
	mutualinfos = np.array([
		binset.mutualinfo("Label")
		for binset in binsets
	])
	maxindex = np.argmax(mutualinfos)
	return (remaining_cols[maxindex], mutualinfos[maxindex], binsets[maxindex])

col, mutualinfo, binset = find_greatest_mutualinfo(traindata, featureCols[:])
print col, mutualinfo

dtree = DecisionTree(binset, "Label")

write("loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

decisions = dtree.decide(testdata)
print "s:", len(filter(lambda item: item=="s", decisions))
print "b:", len(filter(lambda item: item=="b", decisions))
# print "\n" + "\n".join(decisions)

writeDone(time.time() - global_start)