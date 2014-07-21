import numpy as np
import pandas as pd
from bkputils import *
import math, time, sys, zipfile, json

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None
CSV_OUTPUT_FILE = "decisionTree2.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

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
		self.numbins = numbins = int(math.sqrt(math.sqrt(npData.size)))
		if self.numbins == 0:
			self._hist = np.empty(0)
			self._bins = np.empty(0)
		else:
			bindices = np.linspace(0, npData.size-1, numbins+1).astype(np.int32) # bin + indices = bindices! lol I'm hilarious...
			try:
				bins = sortedData[bindices]
			except:
				print
				print sortedData
				print bindices
				print col
				print dataframe
				exit()

			#generate histogram density
			(self._hist, self._bins) = np.histogram(npData, bins, density=True)
			# self._maxval = self._bins[-1]
			# self._maxbindex = numbins # don't subtract one - numpy thinks bins are 1-based not 0-based

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
		if contains_empty_group(self.original_groups):
			print_groups(self.original_groups)
			exit()

	def group(self, dataframe):
		bindices = self._getbindices(dataframe[self.col])
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

		# #account for a minor inconsistency in np.digitize: final bin should
		# #include its upper limit instead of being treated as out-of-bounds
		# result[datapoints == self._maxval] = self._maxbindex

		#anything with bindex 0 or numbins+1 is outside the range seen in the training
		#round them to the closest bin
		result[result == 0] = 1
		result[result == self.numbins+1] = self.numbins

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

		self.original_groups = groups = dataframe.groupby(col)

		self.bins = bins = {}
		for key, group in groups:
			bins[key] = float(group.shape[0]) / dataframe.shape[0]
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
			groups = dataframe.groupby([self.col])

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

class DecisionTree:
	def __init__(self, binset, decisioncol, remaining_cols, depth=0):
		self.binset = binset
		self.depth = depth

		#decisioncol is the column we're making *about* (e.g. the class
		#we're guessing, like "Label")
		self.decisioncol = decisioncol

		#col is the column we're using to inform our guess at this stage of the tree
		#(eg "DER_mass_mmc")
		self.col = self.binset.col

		#remaining_cols is feature_cols without col, or any of the cols of
		#the parent trees
		self.remaining_cols = remaining_cols

		self.children = {}
		for key, group in binset.original_groups:
			if group.shape[0] == 0:
				print_groups(binset.original_groups)
				exit()
			self.children[key] = DecisionLeaf(self, key, group, decisioncol, remaining_cols)

	def to_json_dict(self):
		result = {}
		for key in self.children:
			result[key] = self.children[key].to_json_dict()
		return result

	def decide(self, rows):
		rows["__ungrouped_index__"] = range(rows.shape[0])
		groupkeys = self.binset.groupkeys(rows[self.col])
		rowgroups = rows.groupby(groupkeys)

		decisions = [None] * rows.shape[0]
		confidence = [None] * rows.shape[0]
		for groupkey, rowgroup in rowgroups:
			# if groupkey not in self.children:
			# 	continue
			groupdecisions, groupconfidence = self.children[groupkey].decide(rowgroup)

			for i, rowindex in enumerate(rowgroup["__ungrouped_index__"]):
				decisions[rowindex] = groupdecisions[i]
				confidence[rowindex] = groupconfidence[i]

		return (decisions, confidence)

	def printChildren(self):
		print
		for i in xrange(1, self.binset.numbins+1):
			print "%03d: %f - %f\t%d\t%f" % (
				i,
				self.binset.delegate._hist[i],
				self.binset.delegate._hist[i+1],
				self.children[i].dataframe.shape[0],
				self.children[i]._get_next_mutualinfo()
			)
		# for index, key in enumerate(self.children):
		# 	sys.stdout.write("%s: %s\t" % (str(key), str(self.children[key])))
		# 	if index % 5 == 4:
		# 		print
		# sys.stdout.flush()

	def find_best_leaf(self):
		maxscore = 0.0
		maxleaf = None
		for childkey in self.children:
			mutinf_tuple = self.children[childkey].find_best_leaf()
			score, mutinf, leaf = mutinf_tuple
			if score > maxscore:
				maxscore = score
				maxleaf = mutinf_tuple
		return maxleaf

class DecisionLeaf:
	def __init__(self, parent, parent_key, dataframe, decisioncol, remaining_cols):

		if dataframe.shape[0] == 0:
			raise Exception("foo")

		groups = dataframe.groupby(decisioncol)

		maxkey = None
		maxsize = 0
		for key, data in groups:
			if data.shape[0] > maxsize:
				maxkey = key
				maxsize = data.shape[0]

		#temp
		self.s_size = groups.get_group("s").shape[0] if "s" in groups.groups else 0
		self.b_size = groups.get_group("b").shape[0] if "b" in groups.groups else 0
		self.total_size = dataframe.shape[0]


		self.num_misclassified = dataframe.shape[0] - groups.get_group(maxkey).shape[0]

		self.parent = parent
		self.parent_key = parent_key
		self.decision = maxkey
		self.confidence = float(groups.get_group(maxkey).shape[0]) / dataframe.shape[0]
		self.dataframe = dataframe
		self.decisioncol = decisioncol
		self.remaining_cols = remaining_cols

		#we will compute these later, as needed. Nice to have them initialzed though (easer to check against None)
		self.next_col, self.next_mutualinfo, self.next_binset = (None, None, None)

	def __str__(self):
		return "DecisionLeaf(%s)" % str(self.decision)

	def decide(self, rows):
		return (
			[self.decision] * len(rows),
			[self.confidence] * len(rows)
		)

	def _get_self_stats(self):
		if self.next_col is None:
			self.next_col, self.next_mutualinfo, self.next_binset = \
				find_greatest_mutualinfo(self.dataframe, self.remaining_cols)
			self.growth_potential = self.next_mutualinfo * self.dataframe.shape[0]
			# self.growth_potential = self.num_misclassified
		return (
			self.next_mutualinfo,
			self.growth_potential,
			self.next_col,
			self.next_binset
		)

	def find_best_leaf(self):
		mi, growth_potential, __, __ = self._get_self_stats()
		return (growth_potential, mi, self)

	def to_tree(self):
		muting, growth_potential, next_col, next_binset = self._get_self_stats() # make sure stuff is initialized (should be, but w/e...)
		return DecisionTree(
			next_binset,
			self.decisioncol,
			filter(lambda x: x != next_col, self.remaining_cols),
			depth = self.parent.depth + 1
		)

	def to_json_dict(self):
		return "%s (%04f%%: s*%d + b*%d = %d)" % (
			self.decision,
			self.confidence*100,
			self.s_size,
			self.b_size,
			self.total_size
		)


def print_groups(groups):
	for key, group in groups:
		print str(key) + ":"
		print group

def contains_empty_group(groups):
	for key, group in groups:
		if group.shape[0] == 0:
			return True
	return False

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

# def iterate_tree_leaves(tree):
# 	if isinstance(tree, DecisionLeaf):
# 		yield tree
# 	else:
# 		for subtree in tree.children:
# 			for leaf in iterate_tree_leaves(subtree):
# 				yield leaf

# def find_leaf_with_greatest_mutual_info(tree):
# 	max_mutinf = 0.0
# 	max_leaf = None
# 	for leaf in iterate_tree_leaves(tree):
# 		mutinf = leaf.mutualinfo()


write("making decision tree")
col, mutualinfo, binset = find_greatest_mutualinfo(traindata, featureCols[:])
dtree = DecisionTree(binset, "Label", filter(lambda x: x!=col, featureCols))

print

# print json.dumps(dtree.to_json_dict(), sort_keys=True, indent=4)
# print "-----------------------------"

for i in xrange(10):
	write("iteration %d" % i)
	score, mutualinfo, leaf = dtree.find_best_leaf()
	leaf.parent.children[leaf.parent_key] = leaf.to_tree()
	writeDone()

print json.dumps(dtree.to_json_dict(), sort_keys=True, indent=4)
exit()

write("loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

write("applying decision tree")
decisions, confidence = dtree.decide(testdata)
writeDone()

write("writing output")
testdata["Class"] = decisions
testdata["confidence"] = confidence
testdata = testdata.sort("confidence")
testdata["RankOrder"] = range(1, testdata.shape[0] + 1)
testdata = testdata.sort("EventId")
testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
zf.write(CSV_OUTPUT_FILE)
zf.close()

writeDone()

writeDone(time.time() - global_start)