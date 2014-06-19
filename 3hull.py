import numpy as np
import bottleneck as bn
import pandas as pd
from bkputils import *
import math, time, sys, random, scipy.spatial, itertools
# import cProfile

import matplotlib.pyplot as pyplot
import matplotlib as mpl
from matplotlib.collections import PolyCollection

global_start = time.time()

TRAIN_LIMIT = 10000
TEST_LIMIT = None
DIMENSIONS_PER_CLASSIFIER = 2
NEIGHBORHOOD_DEPTH = None # if none will calc from num_rows

seed = 49
random.seed(seed)
np.random.seed(seed)

def z_score(col):
	"""Calculate the z-scores of a column or pandas.Series"""
	mean = col.mean()
	std = col.std()
	return (col - mean)/std

def z_scores(df):
	"""Calculate the z-scores of a dataframe on a column-by-column basis"""
	return df.apply(z_score, 0)

# def find_neighbors(pindex, triang):
# 	"""
# 	Given an index of a vertex in a scipy.spatial.Delaunay triangulation,
# 	return the indices of all neighboring vertices. This is not to be
# 	confused with finding all of the neighboring simplices of a given
# 	simplex, for example, which is a quite different task
# 	"""

# 	# http://stackoverflow.com/questions/12374781/how-to-find-all-neighbors-of-a-given-point-in-a-delaunay-triangulation-using-sci#answer-23700182
# 	(indices, indptr) = triang.vertex_neighbor_vertices
# 	try:
# 		return indptr[indices[pindex]:indices[pindex+1]]
# 	except:
# 		print
# 		print "pindex:", pindex
# 		print "len(indices):", len(indices)
# 		print "indices[pindex]:", indices[pindex]
# 		# print "indices[pindex+1]:", indices[pindex+1]
# 		print "len(indptr):", len(indptr)
# 		exit()

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

def get_simplex_neighbors(simplex_index, dtri, depth=1, return_depth_neighbors=False):
	all_neighbors = np.array([simplex_index])
	depth_neighbors = [[simplex_index]]
	for i in xrange(1, depth+1):
		set_neighbors = get_simplex_set_neighbors(all_neighbors, dtri)
		depth_neighbors.append(set_neighbors)
		# all_neighbors += list(set_neighbors)
		all_neighbors = np.concatenate((all_neighbors, set_neighbors))

	if return_depth_neighbors:
		return (all_neighbors, depth_neighbors)
	else:
		return all_neighbors

def get_simplex_neighbors2(simplex_index, dtri, depth=1, return_depth_neighbors=False):
	prev_neighbors = np.array([simplex_index])
	depth_neighbors = [prev_neighbors]
	for i in xrange(depth):
		next_neighbors = dtri.neighbors[prev_neighbors].flatten()
		next_neighbors = next_neighbors[next_neighbors != -1]
		next_neighbors = np.setdiff1d(next_neighbors, prev_neighbors)
		depth_neighbors.append(next_neighbors)
		prev_neighbors = next_neighbors
	return np.concatenate(depth_neighbors)

def get_simplex_neighbors3(simplex_index, dtri, depth=1, return_depth_neighbors=False):
	num_simplices = dtri.simplices.shape[0]
	indices = np.arange(0, num_simplices)
	unseen = np.ones(num_simplices, dtype=bool) # array of True

	prev_neighbors = np.array([simplex_index])
	unseen[prev_neighbors] = False
	for i in xrange(depth):
		next_neighbors = dtri.neighbors[prev_neighbors].flatten()
		next_neighbors = next_neighbors[next_neighbors != -1]
		next_neighbors = next_neighbors[unseen[next_neighbors]]
		next_neighbors = np.unique(next_neighbors)
		unseen[next_neighbors] = False
		prev_neighbors = next_neighbors
	# return np.nonzero(~unseen)
	return indices[~unseen]

def get_simplex_set_neighbors(simplex_indexes, dtri):
	# return get_simplex_set_neighbors_bool_array(simplex_indexes, dtri)
	return get_simplex_set_neighbors_npsets(simplex_indexes, dtri)

def get_simplex_set_neighbors_npsets(simplex_indexes, dtri):
	neighbor_indices = dtri.neighbors[simplex_indexes].flatten()
	neighbor_indices = neighbor_indices[neighbor_indices != -1]
	neighbor_indices = np.setdiff1d(neighbor_indices, simplex_indexes)
	return neighbor_indices

def get_simplex_set_neighbors_bool_array(simplex_indexes, dtri):

	num_simplices = dtri.simplices.shape[0]
	is_neighbor = np.zeros(num_simplices, dtype=bool) # array of False

	# simples_indices_list = list(simplex_indexes)
	neighbor_indices = dtri.neighbors[simplex_indexes].flatten()
	neighbor_indices = neighbor_indices[neighbor_indices != -1]
	is_neighbor[neighbor_indices] = True
	is_neighbor[simplex_indexes] = False
	return np.flatnonzero(is_neighbor)


	# set_neighbors = dtri.neighbors[simplex_indexes]
	# # is_new = np.ones(set_neighbors.size, dtype=bool) # initialize bool array of True's
	# # is_new[simplex_indexes] = False
	# # set_neighbors = np.setdiff1d(set_neighbors, simplex_indexes)
	# return set_neighbors[is_new]

def get_simplex_set_vertices(simplex_indexes, dtri):
	all_vertices = dtri.simplices[simplex_indexes].flatten()
	all_vertices = all_vertices[all_vertices != -1]
	return np.unique(all_vertices)

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT)
feature_cols = featureCols()
rand_cols = random.sample(feature_cols, DIMENSIONS_PER_CLASSIFIER)
# rand_cols = feature_cols # testing with all features
writeDone()

write("calculating z-scores")
zscores = z_scores(traindata[rand_cols])
# zscores = zscores.applymap(lambda x: 0. if np.isnan(x) else x)
zscores = zscores.dropna(how="any")
num_rows= zscores.shape[0]
writeDone()
print "num_rows:", num_rows

if NEIGHBORHOOD_DEPTH == None:
	NEIGHBORHOOD_DEPTH = int(math.ceil(2*math.log(num_rows)))
	print "NEIGHBORHOOD_DEPTH:", NEIGHBORHOOD_DEPTH

write("calculating delaunay triangulation")
dt = scipy.spatial.Delaunay(zscores)
num_simplices = dt.simplices.shape[0]
writeDone()

write("calculating simplex volumes")
dt.simplex_volumes = simplex_volumes(dt)
writeDone()

# write("estimating simplex probability densities")
# point_simplex_lookup = [[] for i in xrange(0, num_rows)]
# for simplex_index in xrange(0, num_simplices):
# 	simplex = dt.simplices[simplex_index]
# 	for point_index in simplex:
# 		point_simplex_lookup[point_index].append(simplex_index)
# point_volume_estimates = np.empty(num_rows)
# point_volume_estimates.fill(np.nan)
# for point_index in xrange(0, num_rows):
# 	neighboring_simplex_indexes = point_simplex_lookup[point_index]
# 	neighboring_simplex_volumes = dt.simplex_volumes[neighboring_simplex_indexes]
# 	neighboring_simplex_volume = np.sum(neighboring_simplex_volumes)

# 	num_neighboring_points = len(find_neighbors(point_index, dt))
# 	point_volume_estimates[point_index] = neighboring_simplex_volume / num_neighboring_points
# writeDone()

write("estimating simplex probability densities")
simplex_densities = np.empty(num_simplices)
def doStuff():
	global simplex_densities
	for i in xrange(0, num_simplices):
		neighborhood_simplex_indices = get_simplex_neighbors3(i, dt, depth=NEIGHBORHOOD_DEPTH)
		neighborhood_volume = np.sum(dt.simplex_volumes[neighborhood_simplex_indices])
		simplex_densities[i] = neighborhood_simplex_indices.shape[0] / (neighborhood_volume * num_rows)
	density_normalizer = np.sum(simplex_densities * dt.simplex_volumes)
	simplex_densities = simplex_densities / density_normalizer
# cProfile.run("doStuff()", "3hull.profile")
doStuff()
writeDone()

global_elapsed = time.time() - global_start
print "Took %s" % fmtTime(global_elapsed)

print "plotting %s vs %s..." % (rand_cols[0], rand_cols[1])

pyplot.ion()
fig, ax = pyplot.subplots()
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
coll = PolyCollection(
	dt.points[dt.simplices],
	array=simplex_densities,
	cmap=mpl.cm.jet,
	edgecolors="none"
)
ax.add_collection(coll)
fig.colorbar(coll, ax=ax)
pyplot.xlabel(rand_cols[0])
pyplot.ylabel(rand_cols[1])
pyplot.show()
_ = raw_input("Press Enter to Exit...")