import numpy as np

def calc_col_flags(df, feature_cols):
	row_sums = np.zeros(df.shape[0], dtype=np.int)
	for index, col in enumerate(feature_cols):
		series = df[col]
		row_sums += np.logical_not(np.isnan(series)) * (2**index)
	return row_sums

def get_flagged_cols(col_flags, available_cols):
	return [
		col for (index, col) in enumerate(available_cols)
		if col_flags & 2**index > 0
	]