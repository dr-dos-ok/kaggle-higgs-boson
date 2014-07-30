import pandas as pd
import numpy as np
from bkputils import *

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std(ddof=0)
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, axis=0)

def truth_combinations(n):
	if n == 1:
		yield (False,)
		yield (True,)
	else:
		for com in truth_combinations(n-1):
			yield (False,) + com
		for com in truth_combinations(n-1):
			yield (True,) + com

def truth_combination_to_int(com):
	result = 0
	for index, b in enumerate(com):
		if b:
			place = len(com) - index - 1
			result += 2**place
	return result

def int_to_truth_combination(i, places):
	result = [False] * places
	for place in range(places):
		index = places - place - 1
		if i % 2 == 1:
			result[index] = True
		i = i >> 1
	return tuple(result)

class Partition(object):
	def __init__(self, points, min_corner, max_corner, include_max=None):
		self.min_corner = min_corner
		self.max_corner = max_corner
		
		if include_max is None:
			include_max = np.ones(points.shape[1], dtype=np.bool)
		self.include_max = include_max

		self.points = self.filter(points)
		self.npoints, self.ndim = self.points.shape

	def volume(self):
		return np.prod(np.fabs(self.max_corner - self.min_corner))

	def density(self):
		return self.npoints / self.volume()

	def is_in_partition(self, points):
		gt_min = points >= self.min_corner
		lt_max = points < self.max_corner
		eq_max = self.include_max & (points == self.max_corner)

		return np.all(gt_min & (lt_max | eq_max), axis=1)

	def filter(self, points):
		inside = self.is_in_partition(points)
		return points[inside]

	def split(self):
		midpoint = (self.max_corner + self.min_corner) / 2.0

		result = []
		for index_tuple in truth_combinations(self.ndim):
			sub_min = np.empty(self.ndim)
			sub_max = np.empty(self.ndim)
			include_max = np.zeros(self.ndim, dtype=np.bool)
			for dim_index, is_max in enumerate(index_tuple):
				if is_max:
					sub_min[dim_index] = midpoint[dim_index]
					sub_max[dim_index] = self.max_corner[dim_index]
					include_max[dim_index] = self.include_max[dim_index]
				else:
					sub_min[dim_index] = self.min_corner[dim_index]
					sub_max[dim_index] = midpoint[dim_index]
			sub_partition = Partition(
				self.points,
				sub_min, sub_max,
				include_max=include_max
			)
			result.append(sub_partition)
		return result


if __name__ == "__main__":
	# print int_to_truth_combination(1, 3)
	# for i in range(8):
	# 	print i, int_to_truth_combination(i, 3)
	for index, com in enumerate(truth_combinations(3)):
		print str(index) + ":", com, truth_combination_to_int(com), int_to_truth_combination(truth_combination_to_int(com), 3)