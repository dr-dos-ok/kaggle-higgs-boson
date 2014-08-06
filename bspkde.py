import pandas as pd
import numpy as np
from bkputils import *

import math

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

# [[1],[2],[3]]
# 
# [
#	[[4,1], [4,2], [4,3]],
#	[[5,1], [5,2], [5,3]],
#	[[6,1], [6,2], [6,3]]
# ]

def linspace_nd(min_corner, max_corner, num_per_dim):

	num_dim = len(min_corner)

	linspaces = [
		np.linspace(min_corner[index], max_corner[index], num=num_per_dim)
		for index in xrange(num_dim)
	]

	sides = np.meshgrid(*linspaces)

	coord_indexes = [range(num_per_dim)]*num_dim
	for coord in array_combinations(coord_indexes):
		result = []
		for i in xrange(num_dim):
			result.append(sides[i][coord])
		yield tuple(result)

def array_combinations(arrays):
	arr = arrays[0]
	if len(arrays) == 1:
		for item in arr:
			yield (item,)
	else:
		for sub_item in array_combinations(arrays[1:]):
			for item in arr:
				yield (item,) + sub_item

def _linspace_combinations(sides):
	side = sides[0]
	if len(sides) == 1:
		for item in side:
			yield (item,)
	else:
		for item in side:
			for sub_item in _linspace_combinations(sides[1:]):
				yield (item,) + sub_item

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

class PartitionSet(object):
	def __init__(self, points):

		self.points = points
		self.npoints, self.ndim = points.shape
		self.min_corner = np.min(points, axis=0)
		self.max_corner = np.max(points, axis=0)

		self.partitions = [Partition(points, self.min_corner, self.max_corner)]

	def train(division_cutoff=None, max_depth=None):
		sqrt = int(math.floor(math.sqrt(self.npoints)))
		if division_cutoff is None:
			division_cutoff = sqrt
		if max_depth is None:
			max_depth = sqrt

		final_partitions = []

		done = False
		remaining_partitions = self.partitions
		depth = 0
		while len(remaining_partitions) > 0 and depth <= max_depth:
			next_iteration = []
			for partition in remaining_partitions:
				if partition.count < division_cutoff:
					final_partitions.append(partition)
				else:
					next_iteration += partition.split()
			remaining_partitions = next_iteration
			depth += 1
		self.partitions = final_partitions
		self.depth = depth

		bins_per_dim = 2**(depth-1)

		density_matrix = np.empty([bins_per_dim] * self.ndim)

