import numpy as np
import pandas as pd
from bkputils import *
import math, time, sys, zipfile, json, random, scipy.spatial

global_start = time.time()

TRAIN_LIMIT = None
TEST_LIMIT = None
CSV_OUTPUT_FILE = "multibayes.csv"
ZIP_OUTPUT_FILE = CSV_OUTPUT_FILE + ".zip"

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std()
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, 0)

def calc_col_flags(df):
	row_sums = np.zeros(df.shape[0], dtype=np.int)
	for index, col in enumerate(feature_cols):
		series = df[col]
		row_sums += np.logical_not(np.isnan(series)) * (2**index)
	return row_sums

def get_flagged_cols(col_flags, available_cols = None):
	global feature_cols
	if available_cols == None:
		available_cols = feature_cols
	return [
		col for (index, col) in enumerate(available_cols)
		if col_flags & 2**index > 0
	]

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
	def __init__(self, name, dataframe, target_cols):

		self.name = name

		# write("%s: randomly sampling bin points" % name)
		num_bins = int(math.sqrt(dataframe.shape[0]))
		bin_indices = pd.Series(random.sample(dataframe.index, num_bins))
		# writeDone()
		# print "	%d points, %d bins" % (dataframe.shape[0], num_bins)

		# write("%s: calculating z-scores" % name)
		zscores = self.z_scores(dataframe[target_cols], True)
		self.bin_zscores = bin_zscores = zscores.ix[bin_indices]
		# writeDone()

		# write("%s: calculating delaunay triangulation" % name)
		dt = scipy.spatial.Delaunay(bin_zscores)
		num_simplices = dt.simplices.shape[0]
		# writeDone()
		# print "	num_simplices:", num_simplices

		# # write("%s: approximating bin volumes" % name)
		# """
		# 	Haven't found an easy/fast way to calculate voronoi cell volumes. The
		# 	union of neighboring simplex should be a reasonable estimate. It will
		# 	overshoot by a ratio of approximately 2**n where n=len(target_cols)=dimensions
		# 	but that should normalize out when we normalize the probability densities
		# """
		# vertex_neighbor_simplices = [[] for i in xrange(num_bins)]
		# for simplex_index, simplex in enumerate(dt.simplices):
		# 	for point_index in simplex:
		# 		vertex_neighbor_simplices[point_index].append(simplex_index)
		# simplex_volumes = calc_simplex_volumes(dtri=dt)
		# bin_volumes = np.array([
		# 	sum(simplex_volumes[neighbor_simplices])
		# 	for neighbor_simplices in vertex_neighbor_simplices
		# ])
		# # writeDone()

		# write("%s: calculating voronoi cell volumes" % name)
		indices, indptr = dt.vertex_neighbor_vertices
		def voronoi_cell_volume(point_index):
			neighborhood_indices = indptr[indices[point_index]:indices[point_index+1]]
			neighborhood_points = np.concatenate((
				dt.points[neighborhood_indices],
				dt.points[[point_index]]
			))
			ch = scipy.spatial.ConvexHull(neighborhood_points)
			simplices = np.column_stack((
				np.repeat(ch.vertices[0], ch.nsimplex),
				ch.simplices
			))
			return np.sum(calc_simplex_volumes(simplices=dt.points[simplices]))
		bin_volumes = np.array([
			voronoi_cell_volume(bin_index)
			for bin_index in xrange(num_bins)
		])
		# writeDone()

		# write("%s: sorting points into bins" % name)
		self.kdtree = kdtree = scipy.spatial.cKDTree(bin_zscores)
		__, nearest_neighbor_index = kdtree.query(zscores)
		nearest_neighbor_index = pd.Series(nearest_neighbor_index)
		bin_counts = np.zeros(num_bins)
		for bin_index, group in nearest_neighbor_index.groupby(nearest_neighbor_index):
			bin_counts[bin_index] = group.size
		# writeDone()

		# write("%s: calculating bin densities" % name)
		self.bin_densities = bin_counts / bin_volumes
		self.bin_densities = self.bin_densities / np.sum(self.bin_densities * bin_volumes)
		# writeDone()

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
		lookup = np.array(["b", "s"])
		score_ratio = score_s/score_b
		return (
			lookup[(score_ratio > 1.0).astype(np.int32)],
			score_ratio
		)

def ams(predictions, actual_df):
	predicted_signal = predictions == "s"
	actual_signal = actual_df["Label"] == "s"
	actual_background = ~actual_signal

	s = true_positives = np.sum(actual_df[predicted_signal & actual_signal]["Weight"])
	b = false_positives = np.sum(actual_df[predicted_signal & actual_background]["Weight"])
	B_R = 10.0

	radicand = 2.0 * ((s+b+B_R) * math.log(1.0 + (s/(b+B_R))) - s)

	if radicand < 0.0:
		print "radicand is less than 0, exiting"
		exit()
	else:
		return math.sqrt(radicand)

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
traindata = traindata[:TRAIN_LIMIT]
feature_cols = featureCols(only_float64=True)
target_cols = [
	"DER_mass_MMC",
	"DER_mass_transverse_met_lep",
	"DER_mass_vis",
	"PRI_lep_phi",
	"PRI_met_phi",
	"PRI_tau_phi"
]
writeDone()
print "	num rows:", traindata.shape[0]
print

comparator_lookup = {}
traindata["row_col_flags"] = calc_col_flags(traindata)
for col_flags, group in traindata.groupby("row_col_flags"):
	write("building VoronoiKdeComparator for {0:b}".format(col_flags))
	comparator_lookup[col_flags] = comparator = VoronoiKdeComparator(
		"{0:b}".format(col_flags),
		group,
		get_flagged_cols(col_flags, available_cols=target_cols)
		# get_flagged_cols(col_flags)
	)
	writeDone()
	to_pct = 100.0 / (comparator.num_s + comparator.num_b)
	print "	[s: %02f%%, b: %02f%%]" % (comparator.num_s * to_pct, comparator.num_b * to_pct)
print

def score_df(df):
	df["row_col_flags"] = calc_col_flags(df)
	df["Class"] = ["MONKEY"] * df.shape[0]
	df["confidence"] = ["MONKEY"] * df.shape[0]
	for col_flags, group in df.groupby("row_col_flags"):
		if col_flags in comparator_lookup:
			comparator = comparator_lookup[col_flags]
			class_, confidence = comparator.classify(group)
			df["Class"][group.index] = class_
			df["confidence"][group.index] = confidence
	df = df.sort("confidence")
	df["RankOrder"] = range(1, df.shape[0] + 1)
	df = df.sort("EventId")
	return df

write("classifying training rows")
traindata = score_df(traindata)
writeDone()

print "AMS:", ams(traindata["Class"], traindata)
print "% signal (actual):", (float(np.sum(traindata["Label"]=="s")) / np.sum(traindata.shape[0]))
print "% signal (predicted):", (float(np.sum(traindata["Class"]=="s")) / np.sum(traindata.shape[0]))

# write("loading test data")
# testdata = loadTestData(TEST_LIMIT)
# writeDone()

# write("classifying test data")
# testdata = score_df(testdata)
# writeDone()

# write("writing output")
# testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
# zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
# zf.write(CSV_OUTPUT_FILE)
# zf.close()
# writeDone()

writeDone(time.time() - global_start)