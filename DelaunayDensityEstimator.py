import numpy as np
from scipy.spatial import Delaunay
import cPickle as pickle
import math, os

SAVE_DIR = "ddes/"
IMG_SAVE_DIR = "dde_imgs/"

def default_name(columns, num_rows):
	return "%s|%s" % (
		"|".join(columns),
		num_rows
	)

def default_save_name(columns, num_rows):
	return SAVE_DIR + default_name(columns, num_rows) + ".dde"

def default_img_name(columns, num_rows):
	return IMG_SAVE_DIR + default_name(columns, num_rows) + ".png"

def issaved(columns, num_rows):
	return os.path.isfile(default_save_name(columns, num_rows))

def load(columns, num_rows):
	with open(default_save_name(columns, num_rows), "r") as infile:
		return pickle.load(infile)

class DelaunayDensityEstimator:
	def __init__(self, data, neighborhood_depth=None, profile=None):

		# normalize data using z-scores
		# be sure to save off mean and stddev so that we
		# can perform the same transform on test data later
		self.means = data.mean(axis=0)
		self.stddevs = data.std(axis=0)
		self.normalized_data = (data - self.means) / self.stddevs
		self.normalized_data.dropna(how="any")
		(self.num_rows, self.num_dim) = self.normalized_data.shape

		self.columns = list(self.normalized_data.columns.values)

		# unless a neighborhood depth was provided, calculate one using
		# a heuristic formula: (1/2) * nthroot(num_rows, num_dim+1).
		# This formula has absolutely no logic behind it, except that it
		# seems to come up with half-decent numbers for the datasets I'm
		# using (1k-250k rows of training data in 2D)
		#
		# The more data you want to train on, the more you have to expand
		# the size of your neighborhood to limit the small random pockets of 
		# of very high density that appear and muck everything up. Since
		# the runtime of the density estimation routine is roughly
		# O(num_rows * items_per_neighborhood) and items_per_neighborhood
		# increases rapidly with neighborhood depth (O(depth^num_dim), I think)
		# the runtime gets tedious for large (>100k) training sets pretty
		# easily
		if neighborhood_depth != None:
			self._NEIGHBORHOOD_DEPTH = neighborhood_depth
		else:
			dimth_root = math.pow(self.num_rows, 1.0 / (self.num_dim+1))
			dimth_root /= 2.0
			self._NEIGHBORHOOD_DEPTH = int(math.ceil(dimth_root))

		# calc Delaunay triangulation
		self.dt = dt = Delaunay(self.normalized_data)
		self.num_simplices = dt.simplices.shape[0]

		# calc simplex volume and estimate their densities
		self.simplex_volumes = self._calc_simplex_volumes()
		if profile == None:
			self.simplex_densities = self._estimate_simplex_densities()
		else:
			import cProfile
			cProfile.runctx("self.simplex_densities = self._estimate_simplex_densities()", globals(), locals(), filename=profile)

	def _calc_simplex_volumes(self):
		"""
		Given a scipy.spatial.Delaunay object, calculate the n-dimensional
		(unsigned) volume of each simplex and return that amount in a 
		numpy.ndarray that is synchronized to the Delaunay.simplices array
		of the input object
		"""

		dtri = self.dt

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

	def _estimate_simplex_densities(self):
		simplex_densities = np.empty(self.num_simplices)
		simplex_volumes = self.simplex_volumes
		for i in xrange(0, self.num_simplices):
			neighborhood_simplex_indices = self.get_simplex_neighbors(i, self._NEIGHBORHOOD_DEPTH)
			num_neighbors = neighborhood_simplex_indices.shape[0]
			neighborhood_volume = np.sum(simplex_volumes[neighborhood_simplex_indices])
			simplex_densities[i] = num_neighbors / neighborhood_volume
		normalizing_constant = np.sum(simplex_densities * simplex_volumes)
		simplex_densities = simplex_densities / normalizing_constant
		return simplex_densities

	def get_simplex_neighbors(self, simplex_index, depth):
		dtri = self.dt
		num_simplices = dtri.simplices.shape[0]
		indices = np.arange(0, num_simplices)
		unseen = np.ones(num_simplices, dtype=bool) # array of True; all points are unseen to start with

		prev_neighbors = np.array([simplex_index])
		unseen[prev_neighbors] = False
		for i in xrange(depth):
			next_neighbors = dtri.neighbors[prev_neighbors].flatten()
			next_neighbors = next_neighbors[next_neighbors != -1]
			next_neighbors = next_neighbors[unseen[next_neighbors]]
			# next_neighbors = np.unique(next_neighbors)
			unseen[next_neighbors] = False
			prev_neighbors = next_neighbors

		# return np.nonzero(~unseen)
		return indices[~unseen]

	def score(self, testdata):
		normalized_testdata = (testdata - self.means) / self.stddevs
		simplex_indices = self.dt.find_simplex(normalized_testdata)
		simplex_indices[simplex_indices == -1] = np.na
		return self.simplex_densities[simplex_indices]

	def default_save_name(self, num_rows_override=None):
		num_rows = num_rows_override if num_rows_override != None else self.num_rows
		return default_save_name(self.columns, num_rows)

	def default_img_name(self, num_rows_override=None):
		if not os.path.isdir(IMG_SAVE_DIR):
			os.mkdir(IMG_SAVE_DIR)
		num_rows = num_rows_override if num_rows_override != None else self.num_rows
		return default_img_name(self.columns, self.num_rows)

	def save(self, num_rows_override=None):
		if not os.path.isdir(SAVE_DIR):
			os.mkdir(SAVE_DIR)
		with open(self.default_save_name(num_rows_override), "w") as outfile:
			pickle.dump(self, outfile)