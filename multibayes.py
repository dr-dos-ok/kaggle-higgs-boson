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

def simplex_volumes(dtri):
	"""
	Given a scipy.spatial.Delaunay object, calculate the n-dimensional
	(unsigned) volume of each simplex and return that amount in a 
	numpy.ndarray that is synchronized to the Delaunay.simplices array
	of the input object
	"""

	# dtri.simplices is a 2D array of [simplex_index, point_index] = pointLookup
	# look up actual points to convert this to a 3D array of 
	# [simplex_index, point_index, coordIndex] = coordScalar
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

		write("%s: randomly sampling bin points" % name)
		num_bins = int(math.sqrt(dataframe.shape[0]))
		bin_indices = pd.Series(random.sample(dataframe.index, num_bins))
		bin_points = dataframe.ix[bin_indices]
		writeDone()
		print "	%d points, %d bins" % (dataframe.shape[0], num_bins)

		write("%s: calculating z-scores" % name)
		zscores = self.z_scores(traindata[target_cols], True)
		self.bin_zscores = bin_zscores = zscores.ix[bin_indices]
		writeDone()

		write("%s: calculating delaunay triangulation" % name)
		dt = scipy.spatial.Delaunay(bin_zscores)
		num_simplices = dt.simplices.shape[0]
		writeDone()
		print "	num_simplices:", num_simplices

		write("%s: approximating bin volumes" % name)
		"""
			Haven't found an easy/fast way to calculate voronoi cell volumes. The
			union of neighboring simplex should be a reasonable estimate. It will
			overshoot by a ratio of approximately 2**n where n=len(target_cols)=dimensions
			but that should normalize out when we normalize the probability densities
		"""
		vertex_simplex_lookup = [[] for i in xrange(num_bins)]
		for simplex_index, simplex in enumerate(dt.simplices):
			for point_index in simplex:
				vertex_simplex_lookup[point_index].append(simplex_index)
		bin_volumes = pd.Series([
			sum(vertex_simplex_lookup[index])
			for index in xrange(num_bins)
		])
		# bin_volumes = bin_volumes / sum(bin_volumes)
		writeDone()

		write("%s: sorting points into bins" % name)
		self.kdtree = kdtree = scipy.spatial.cKDTree(bin_zscores)
		__, nearest_neighbor_index = kdtree.query(zscores)
		nearest_neighbor_index = pd.Series(nearest_neighbor_index)
		bin_counts = pd.Series(np.zeros(num_bins))
		for bin_index, group in nearest_neighbor_index.groupby(nearest_neighbor_index):
			bin_counts[bin_index] = group.size
		writeDone()

		write("%s: calculating bin densities" % name)
		self.bin_densities = bin_counts / bin_volumes
		writeDone()

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
		return self.bin_densities[nearest_neighbor_index].values

print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 4 if TRAIN_LIMIT != None else None)
traindata = traindata.dropna(how="any")
traindata = traindata[:TRAIN_LIMIT]
featureCols = featureCols(only_float64=True)
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
print "\n"

traindata_s = traindata[traindata["Label"] == "s"]
traindata_b = traindata[traindata["Label"] == "b"]

kde_s = VoronoiKde("s", traindata_s, target_cols)
print "\n"
kde_b = VoronoiKde("b", traindata_b, target_cols)
print "\n"

write("loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

write("scoring test rows")
score_b = kde_b.score(testdata[target_cols])
score_s = kde_s.score(testdata[target_cols])
score_diff = score_s - score_b
testdata["score_diff"] = score_diff
class_lookup = np.array(["b", "s"])
testdata["Class"] = class_lookup[(score_diff > 0.0).astype(int)]
# testdata["Class"] = ["s" if score > 0.0 else "b" for score in score_diff]
testdata = testdata.sort("score_diff")
testdata["RankOrder"] = range(1, testdata.shape[0] + 1)
testdata = testdata.sort("EventId")
writeDone()

write("writing output")
testdata[["EventId", "RankOrder", "Class"]].to_csv(CSV_OUTPUT_FILE, header=True, index=False)
zf = zipfile.ZipFile(ZIP_OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED)
zf.write(CSV_OUTPUT_FILE)
zf.close()
writeDone()

writeDone(time.time() - global_start)