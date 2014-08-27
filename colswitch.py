import numpy as np

class ColSwitchClassifier(object):
	"""
	This class is a classifier that delegates to a set of
	sub-classifiers. Each row to be classified is analyzed in
	terms of which columns are on/off (off = mising/nan)
	and passed to a classifier that specializes in that
	set of on/off values.

	Internally it uses a "col flag" which is an int where
	each bit represents a column. A 1 in that column means
	the column is available for that row and a 0 means it
	is missing/nan
	"""
	def __init__(self, dataframe, feature_cols, classifier_factory):
		self.lookup = {}
		self.all_feature_cols = feature_cols
		self.nfeatures = len(feature_cols)

		row_col_flags = self.calc_col_flags(dataframe)
		for col_flag, group in dataframe.groupby(row_col_flags):
			self.lookup[col_flag] = classifier_factory(col_flag, self.get_flagged_cols(col_flag), group)

	def calc_col_flags(self, df):
		"""
		For a passed set of data (that is assumed to have the same
		columns as this instance was initialized with) calculate
		the col_flag of each row. The value to indicate a missing
		value is np.nan
		"""
		row_sums = np.zeros(df.shape[0], dtype=np.int)
		for index, col in enumerate(self.all_feature_cols):
			series = df[col]
			row_sums += np.logical_not(np.isnan(series)) * (2**index)
		return row_sums

	def get_flagged_cols(self, col_flag):
		"""
		Return the subset of self.all_feature_cols that this col_flag represents
		"""
		return [
			col for (index, col) in enumerate(self.all_feature_cols)
			if col_flag & 2**index > 0
		]

	def all_on_classifier(self):
		"""
		Return the sub-classifier that represents the case of all
		columns being on
		"""
		return self.lookup[2**(self.nfeatures+1)-1]

	def classify(self, dataframe):
		"""
		Classify a dataframe by splitting out the rows based on colflags,
		and then delegating to the appropriate sub-classifier. Re-assemble
		individual sub-results, and return
		"""
		row_col_flags = self.calc_col_flags(dataframe)
		is_first = True
		for col_flag, group in dataframe.groupby(row_col_flags):
			if col_flag in self.lookup:
				classifier = self.lookup[col_flag]
				class_, confidence = classifier.classify(group)
				if is_first:
					all_classes = np.empty(dataframe.shape[0], dtype=class_.dtype)
					all_confidences = np.empty(dataframe.shape[0], dtype=np.float)
					is_first = False
				all_classes[group.index] = class_
				all_confidences[group.index] = confidence
			else:
				raise Exception("Encountered new colflag in ColSwitchClassifier.classify(). Unable to classify")
		return (all_classes, all_confidences)