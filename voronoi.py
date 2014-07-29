import pandas as pd
import numpy as np
import math, random, scipy.spatial

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std(ddof=0)
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, axis=0)

def calc_simplex_volumes(dtri=None, simplices=None):
	"""
	Given a scipy.spatial.Delaunay object, calculate the n-dimensional
	(unsigned) volume of each simplex and return that amount in a 
	numpy.ndarray that is synchronized to the Delaunay.simplices array
	of the input object
	"""

	# dtri.simplices is a 2D array of [simplex_index, point_index] = pointLookup
	# look up actual points to convert this to a 3D array of 
	# [simplex_index, point_index, coordIndex] = coordScalar
	if simplices is None:
		if dtri is None:
			raise Exception("calc_simplex_volumes requires dtri to be passed")
		simplices = dtri.points[dtri.simplices]

	# simplices in n-space are specified with n+1 points (eg 3 points
	# for a triangle in 2D, 4 points for a tetrahedron in 3D). To
	# simplify the area calculation, substract out one of the points
	# from the others and remove it so that we only have to deal
	# with n points. The calculation then assumes that the missing
	# (n+1)th point is the origin which will be true after the subtraction
	# translates all of the other points
	#
	# Formulas and general info:
	# http://en.wikipedia.org/wiki/Tetrahedron#Volume
	# http://en.wikipedia.org/wiki/Simplex#Volume
	#
	# Source borrowed in large part from:
	# https://github.com/scipy/scipy/blob/0a0409dbe50d4e304b261d3657bd7f5c8580232c/scipy/spatial/tests/test_qhull.py#L106
	# and:
	# http://mail.scipy.org/pipermail/scipy-user/2012-October/033436.html
	shifted = simplices[:,:-1,:] - simplices[:,-1,None,:]

	# extract some useful numbers from the shape.
	# Note: nptsPerSimplex would usually be nptsPerSimplex==ndim+1 but now it will
	# just be nptsPerSimplex==ndim because we stripped out the (n+1)th point above
	(nsimplices, nptsPerSimplex, ndim) = shifted.shape

	# calculate 1/n! (n=3 in 3D, n=4 in 4D, etc). We'll need that constant a lot
	# pretty soon so just get it out of the way now.
	invfact = 1.0 / math.factorial(ndim)

	# use formula nVolume = (1/n!) * det(pts)
	# http://en.wikipedia.org/wiki/Simplex#Volume
	return np.abs(invfact * np.linalg.det(shifted))

class VoronoiKde(object):
	def __init__(self, name, dataframe, target_cols, bin_indices=None):

		self.name = name

		if bin_indices is None:
			num_bins = int(math.sqrt(dataframe.shape[0]))
			bin_indices = pd.Series(random.sample(dataframe.index, num_bins))
			zscores = self.z_scores(dataframe[target_cols], True)
			self.bin_zscores = bin_zscores = zscores.ix[bin_indices]
		else:
			num_bins = len(bin_indices)
			bin_zscores = dataframe.ix[bin_indices]
			zscores = dataframe

		vor = scipy.spatial.Voronoi(bin_zscores)

		def voronoi_cell_volume(point_index):
			neighborhood_indices = vor.regions[vor.point_region[point_index]]
			neighborhood_indices = filter(lambda x: x != -1, neighborhood_indices)

			neighborhood_points = np.concatenate((
				vor.vertices[neighborhood_indices],
				vor.points[[point_index]]
			))
			sub_dt = scipy.spatial.Delaunay(neighborhood_points)
			neighborhood_volume = np.sum(calc_simplex_volumes(dtri=sub_dt))

			return neighborhood_volume

		bin_volumes = np.array([
			voronoi_cell_volume(bin_index)
			for bin_index in xrange(num_bins)
		])

		self.kdtree = kdtree = scipy.spatial.cKDTree(bin_zscores)
		__, nearest_neighbor_index = kdtree.query(zscores)
		nearest_neighbor_index = pd.Series(nearest_neighbor_index)
		bin_counts = np.zeros(num_bins)
		for bin_index, group in nearest_neighbor_index.groupby(nearest_neighbor_index):
			bin_counts[bin_index] = group.size

		self.bin_densities = bin_counts / bin_volumes
		self.bin_densities = self.bin_densities / np.sum(self.bin_densities * bin_volumes)

	def z_scores(self, dataframe, init):
		if init:
			self.stddevs = stddevs = dataframe.std()
			self.means = means = dataframe.mean()
		else:
			means = self.means
			stddevs = self.stddevs
		return (dataframe - means) / stddevs

	def score(self, dataframe):
		zscores = self.z_scores(dataframe, False)
		__, nearest_neighbor_index = self.kdtree.query(zscores)
		return self.bin_densities[nearest_neighbor_index]

class VoronoiKdeComparator(object):
	def __init__(self, name, dataframe, target_cols):

		self.target_cols = target_cols
		self.name = name

		is_s = dataframe["Label"] == "s"
		dataframe_s = dataframe[is_s]
		self.kde_s = VoronoiKde(name + "[s]", dataframe_s, target_cols)
		self.num_s = dataframe_s.shape[0]

		dataframe_b = dataframe[~is_s]
		self.kde_b = VoronoiKde(name + "[b]", dataframe_b, target_cols)
		self.num_b = dataframe_b.shape[0]

	def classify(self, dataframe):
		df = dataframe[self.target_cols]
		score_b = self.kde_b.score(df)
		score_s = self.kde_s.score(df)
		
		score_ratio = score_s/score_b
		return score_ratio

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

class VoronoiKdeComparatorSet(object):
	def __init__(self, col_flags_str, dataframe, cols):
		self.available_cols =  cols

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
				VoronoiKdeComparator(
				"{0:s} ({1:s}, {2:s})".format(col_flags_str, col_pair[0], col_pair[1]),
				dataframe,
				col_pair
				)
			)
		# self.comparator_set = [
		# 	VoronoiKdeComparator(
		# 		"foo", # "{0:b}".format(col_flags),
		# 		dataframe,
		# 		col_pair
		# 	)
		# 	for col_pair in choose(feature_cols, 2)
		# ]

	def classify(self, dataframe):
		score_ratios = np.ones(dataframe.shape[0])
		confidences = np.ones(dataframe.shape[0])
		# print_df = pd.DataFrame()
		for comparator in self.comparator_set:
			sub_score_ratios = comparator.classify(dataframe)
			# print_df[comparator.name] = sub_score_ratios
			score_ratios = score_ratios * sub_score_ratios
		# print print_df
		# exit()
		lookup = np.array(["b", "s"])
		return (
			lookup[(score_ratios > 1.0).astype(np.int)],
			confidences
		)