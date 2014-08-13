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

def choose(n, k):
	if k == 1:
		for item in n:
			yield [item]
	elif len(n) == k:
		yield n
	else:
		current = [n[0]]
		remaining = n[1:]
		for sub_result in choose(remaining, k-1):
			yield current + sub_result
		for sub_result in choose(remaining, k):
			yield sub_result

class Partition(object):
	def __init__(self, name, points, min_corner, max_corner, include_max=None, normalizing_constant=None):

		self.name = name

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
				"(child)",
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

	def get_xlim(self):
		return (self.min_corner[0], self.max_corner[0])

	def get_ylim(self):
		return (self.min_corner[1], self.max_corner[1])

	def title(self):
		return "%s (%d leafs, %d pts)" % (self.name, self.count_leaf_children(), self.points.shape[0])

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

	def score(self, dataframe):
		return self.get_density_estimates(dataframe.values)

class KdeComparator(object):
	def __init__(self, name, dataframe, target_cols):
		
		self.target_cols = target_cols
		self.name = name

		is_s = dataframe["Label"] == "s"
		dataframe_s = dataframe[is_s]
		self.kde_s = self.make_kde("signal", dataframe_s, dataframe)
		self.num_s = dataframe_s.shape[0]
		self.prob_s = float(self.num_s) / dataframe.shape[0]

		dataframe_b = dataframe[~is_s]
		self.kde_b = self.make_kde("background", dataframe_b, dataframe)
		self.num_b = dataframe_b.shape[0]
		self.prob_b = float(self.num_b) / dataframe.shape[0]

	def make_kde(self, name, dataframe):
		raise Exception("You should make a class that subclasses KdeComparator and overrides make_kde()")

	def classify(self, dataframe):
		df = dataframe[self.target_cols]
		score_b = self.kde_b.score(df) * self.prob_b
		score_s = self.kde_s.score(df) * self.prob_s
		
		score_ratio = score_s/score_b
		return score_ratio

	def plot(self):
		pyplot.clf()
		fig, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True, sharey=True)

		max_density = max(self.kde_s.max_density(), self.kde_b.max_density())
		norm = mpl.colors.Normalize(vmin=0.0, vmax=max_density)

		points_s = self.kde_s.plot_heatmap(fig, ax1, norm)
		points_b = self.kde_b.plot_heatmap(fig, ax2, norm)

		fig.suptitle(self.name)
		# ax1.set_title("signal (%d of %d prob=%2.0f%%)" % (self.kde_s.num_bins, self.kde_s.npoints, self.prob_s*100))
		# ax2.set_title("background (%d of %d prob=%2.0f%%)" % (self.kde_b.num_bins, self.kde_b.npoints, self.prob_b*100))
		ax1.set_title(self.kde_s.title())
		ax2.set_title(self.kde_b.title())

		xlim_s = self.kde_s.get_xlim()
		xlim_b = self.kde_b.get_xlim()
		ax1.set_xlim([
			min(xlim_s[0], xlim_b[0]),
			max(xlim_s[1], xlim_b[1])
		])

		ylim_s = self.kde_s.get_ylim()
		ylim_b = self.kde_b.get_ylim()
		ax1.set_ylim([
			min(ylim_s[0], ylim_b[0]),
			max(ylim_s[1], ylim_b[1])
		])

		pyplot.show()
		__ = raw_input("Enter to continue...")
		pyplot.close()

class BspKdeComparator(KdeComparator):
	def __init__(self, name, dataframe, target_cols):
		super(BspKdeComparator, self).__init__(name, dataframe, target_cols)

	def make_kde(self, name, dataframe, superframe):

		min_corner = np.amin(superframe[self.target_cols].values, axis=0)
		max_corner = np.amax(superframe[self.target_cols].values, axis=0)

		diff = max_corner - min_corner
		margin = 0.05 * diff
		max_corner = max_corner + margin
		min_corner = min_corner - margin

		p = Partition(name, dataframe[self.target_cols].values, min_corner, max_corner)
		p.train()

		return p

class ComparatorSet(object):
	def __init__(self, col_flags_str, dataframe, cols):

		self.available_cols = cols
		
		# PRI_jet_all_pt is purely zeroes in some groups where
		# the other jet columns are all -999 (nan). In that case
		# this has zero information to give us, and will cause
		# headaches with the Delaunay Triangulation, so just
		# remove it.
		# 
		# Update: similar logic for DER_pt_tot
		if col_flags_str == "10000001111111110111110001110" or col_flags_str == "10000001111111110111110001111":
			self.available_cols.remove("PRI_jet_all_pt")
			self.available_cols.remove("DER_pt_tot")

		if col_flags_str == "10001111111111110111110001110" or col_flags_str == "10001111111111110111110001111":
			self.available_cols.remove("PRI_jet_all_pt")

		self.comparator_set = []
		for col_pair in choose(self.available_cols, 2):
			self.comparator_set.append(
				self.make_comparator(
					"%s (%s %s)" % (col_flags_str, col_pair[0], col_pair[1]),
					dataframe,
					col_pair
				)
			)

	def make_comparator(self, name, dataframe, cols):
		raise Exception("You should implement this in a subclass")

	def classify(self, dataframe):
		score_ratios = np.ones(dataframe.shape[0])
		confidences = np.ones(dataframe.shape[0])
		for comparator in self.comparator_set:
			sub_score_ratios = comparator.classify(dataframe)
			score_ratios = score_ratios * sub_score_ratios
		lookup = np.array(["b", "s"])
		return (
			lookup[(score_ratios > 1.0).astype(np.int)],
			confidences
		)

	def plot(self):
		for comparator in self.comparator_set:
			comparator.plot()

class BspKdeComparatorSet(ComparatorSet):

	def __init__(self, col_flags_str, dataframe, cols):
		super(BspKdeComparatorSet, self).__init__(col_flags_str, dataframe, cols)

	# override
	def make_comparator(self, name, dataframe, cols):
		return BspKdeComparator(name, dataframe, cols)