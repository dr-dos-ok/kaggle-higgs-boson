import numpy as np
import bottleneck as bn
import pandas as pd
from bkputils import *
import math, time, sys, random, scipy.spatial

global_start = time.time()

random.seed(42)
np.random.seed(42)

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std()
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, 0)

def find_neighbors(pindex, triang):
	"""
	Given an index of a vertex in a scipy.spatial.Delaunay triangulation,
	return the indices of all neighboring vertices. This is not to be
	confused with finding all of the neighboring simplices of a given
	simplex, for example, which is a quite different task
	"""

	# http://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci#answer-23700182
	(indices, indptr) = triang.vertex_neighbor_vertices
	return indptr[indices[pindex]:indices[pindex+1]]

def simplex_volumes(dtri):
	"""
	Given a scipy.spatial.Delaunay object, calculate the n-dimensional
	(unsigned) volume of each simplex and return that amount in a 
	numpy.ndarray that is synchronized to the Delaunay.simplices array
	of the input object
	"""

	# dtri.simplices is a 2D array of [simplexIndex, pointIndex] = pointLookup
	# look up actual points to convert this to a 3D array of 
	# [simplexIndex, pointIndex, coordIndex] = coordScalar
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
	# later on so just get it out of the way now.
	invfact = 1.0 / math.factorial(ndim)

	# use formula nVolume = (1/n!) * det(pts)
	# http://en.wikipedia.org/wiki/Simplex#Volume
	return np.abs(invfact * np.linalg.det(shifted))

TRAIN_LIMIT = 1000
TEST_LIMIT = None
DIMENSIONS_PER_CLASSIFIER = 4

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
featureCols = featureCols()
numrows = traindata.shape[0]
randCols = random.sample(featureCols, DIMENSIONS_PER_CLASSIFIER)
# randCols = featureCols # testing with all features
writeDone()

write("calculating z-scores")
zscores = z_scores(traindata[randCols])
# zscores = zscores.applymap(lambda x: 0. if np.isnan(x) else x)
zscores = zscores.dropna(how="any")
writeDone()

write("calculating delaunay triangulation")
dt = scipy.spatial.Delaunay(zscores)
writeDone()

global_elapsed = time.time() - global_start
print "Took %d:%d" % (global_elapsed/60, global_elapsed%60)