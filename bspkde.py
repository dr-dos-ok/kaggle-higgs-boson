import pandas as pd
import numpy as np
from bkputils import *

import math

import matplotlib.pyplot as pyplot
import matplotlib as mpl
from matplotlib.collections import PolyCollection

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

class Partition(object):
	def __init__(self, points, min_corner, max_corner, include_max=None, normalizing_constant=None):
		self.min_corner = min_corner
		self.max_corner = max_corner
		self.midpoint = (self.max_corner + self.min_corner) / 2.0
		
		if include_max is None:
			include_max = np.ones(points.shape[1], dtype=np.bool)
		self.include_max = include_max

		self.points = self.filter(points)
		self.npoints, self.ndim = self.points.shape

		if normalizing_constant is None:
			normalizing_constant = self.npoints
		self.normalizing_constant = normalizing_constant

		self.children = None

	def volume(self):
		return np.prod(np.fabs(self.max_corner - self.min_corner))

	def density(self):
		return self.npoints / (self.volume() * self.normalizing_constant)

	def is_in_partition(self, points):
		gt_min = points >= self.min_corner
		lt_max = points < self.max_corner
		eq_max = self.include_max & (points == self.max_corner)

		return np.all(gt_min & (lt_max | eq_max), axis=1)

	def filter(self, points):
		inside = self.is_in_partition(points)
		return points[inside]

	def train(self, maxdepth=None, min_pts=None):

		sqrt = int(math.floor(math.sqrt(self.npoints)))
		if min_pts is None:
			min_pts = 2.0 * sqrt # TODO - figure out a better heuristic
		if maxdepth is None:
			maxdepth = sqrt

		if maxdepth <= 0 or self.npoints < min_pts:
			return
		else:
			self.children = self.split()
			for child in self.children:
				child.train(maxdepth=maxdepth-1, min_pts=min_pts)

	def split(self):
		midpoint = self.midpoint

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
				include_max=include_max,
				normalizing_constant = self.normalizing_constant
			)
			result.append(sub_partition)
		return result

	def get_density_estimates(self, pts):
		if self.children is None:
			density = self.density()
			return np.array([density] * pts.shape[0])
		else:
			pt_child_indexes = self.get_child_indices(pts)
			result = np.empty(pts.shape[0], dtype=np.float64)
			for child_index, child in enumerate(self.children):
				mapped_to_child = (pt_child_indexes == child_index)
				return_indices = np.nonzero(mapped_to_child)
				sub_densities = child.get_density_estimates(pts[mapped_to_child])
				result[return_indices] = sub_densities
			return result

	def get_child_indices(self, pts):
		midpoint = self.midpoint

		ndim = midpoint.shape[0]
		results = np.zeros(pts.shape[0], dtype=np.int32)
		for dim_num in range(ndim):
			dim_index = ndim - dim_num - 1
			addend = 1<<dim_num # 1<<x == 2**x
			results[pts[:,dim_index] >= midpoint[dim_index]] += addend
		return results

	def max_density(self):
		if self.children is None:
			return self.density()
		else:
			return max([child.max_density() for child in self.children])

	def count_leaf_children(self):
		if self.children is None:
			return 1 # self
		else:
			return sum([child.count_leaf_children() for child in self.children])

	def plot(self):
		pyplot.clf()
		fig, ax = pyplot.subplots(1, 1)
		norm = mpl.colors.Normalize(vmin=0.0, vmax=self.max_density())

		ax.set_xlim([self.min_corner[0], self.max_corner[0]])
		ax.set_ylim([self.min_corner[1], self.max_corner[1]])
		self.plot_heatmap(fig, ax, norm)

		pyplot.show()

	def _get_polys(self):
		if self.children is None:
			return (
				[[
					[self.min_corner[0], self.min_corner[1]],
					[self.min_corner[0], self.max_corner[1]],
					[self.max_corner[0], self.max_corner[1]],
					[self.max_corner[0], self.min_corner[1]]
				]],
				[self.density()]
			)
		else:
			result_polys = []
			result_densities = []

			for child in self.children:
				child_polys, child_densities = child._get_polys()

				for index in xrange(len(child_densities)):
					result_polys.append(child_polys[index])
					result_densities.append(child_densities[index])

			return (result_polys, result_densities)

	def plot_heatmap(self, fig, ax, norm):
		data, densities = self._get_polys()

		data = np.array(data)
		densities = np.array(densities)

		coll = PolyCollection(
			data,
			array=densities,
			cmap=mpl.cm.jet,
			norm=norm,
			edgecolors="none"
			# linewidths=0.5
		)

		ax.add_collection(coll)
		fig.colorbar(coll, ax=ax)
		# ax.set_xlabel(self.)
		return np.array(data)
