import numpy as np
import cPickle as pickle

import matplotlib as mpl
# mpl.use("Agg") # must come before pyplot
import matplotlib.pyplot as plt

import math, time, random, cProfile, DelaunayDensityEstimator, multiprocessing, os

from bkputils import *
from matplotlib.collections import PolyCollection

global_start = time.time()

TRAIN_LIMIT = 100
TEST_LIMIT = None
USE_MULTIPROCESSING = True

USE_MANUAL_TESTS = False
MANUAL_TESTS = [
	# ["DER_mass_transverse_met_lep", "DER_deltaeta_jet_jet"]
	# ["DER_deltaeta_jet_jet", "PRI_lep_phi"],
	# ["DER_sum_pt", "DER_deltar_tau_lep"],
	# ["DER_mass_transverse_met_lep", "DER_mass_vis"],
	# ["DER_mass_transverse_met_lep", "DER_pt_ratio_lep_tau"],
	# ["DER_mass_transverse_met_lep", "DER_sum_pt"],
	# ["DER_mass_transverse_met_lep", "PRI_lep_eta"],
	# ["DER_mass_transverse_met_lep", "PRI_met_sumet"],
	# ["DER_prodeta_jet_jet", "PRI_lep_eta"],
	# ["PRI_lep_phi", "PRI_met_phi"]
]

seed = 42
random.seed(seed)
np.random.seed(seed)

def pairs(alist):
	num_items = len(alist)
	for i1 in range(0, num_items):
		for i2 in range(i1+1, num_items):
			yield [alist[i1], alist[i2]]

class DdePair:
	SAVE_DIR = "dde_pairs/"
	IMG_SAVE_DIR = "dde_imgs/"

	def __init__(self, dde_s, dde_b, size, cols):
		self.dde_s = dde_s
		self.dde_b = dde_b
		self.size = size
		self.cols = cols

	@staticmethod
	def get_pair(cols, size, data):
		filename = DdePair.default_save_name(cols, size)
		if os.path.isfile(filename):
			from_disk = True
			dde_pair = DdePair._load_pair(cols, size)
		else:
			from_disk = False
			dde_pair = DdePair._make_pair(cols, size, data)
		return (from_disk, dde_pair)

	@staticmethod
	def _load_pair(cols, size):
		filename = DdePair.default_save_name(cols, size)
		with open(filename, "r") as infile:
			return pickle.load(infile)

	@staticmethod
	def _make_pair(cols, size, data):
		dde_data_s = data[data.Label == "s"][cols].dropna(how="any")[:size]
		dde_data_b = data[data.Label == "b"][cols].dropna(how="any")[:size]

		dde_s = DelaunayDensityEstimator.DelaunayDensityEstimator(dde_data_s)
		dde_b = DelaunayDensityEstimator.DelaunayDensityEstimator(dde_data_b)

		return DdePair(dde_s, dde_b, size, cols)

	@staticmethod
	def default_save_name(cols, size):
		return DdePair.SAVE_DIR + str(size) + "/" + ("|".join(cols)) + ".ddepair"

	@staticmethod
	def default_img_name(cols, size):
		return DdePair.IMG_SAVE_DIR + str(size) + "/" + ("|".join(cols)) + ".png"

	def score(self, data):
		score_s = self.dde_s.score(data)
		score_b = self.dde_b.score(data)
		return score_s - score_b

	def save(self):
		if not os.path.isdir(DdePair.SAVE_DIR):
			os.mkdir(DdePair.SAVE_DIR)
		if not os.path.isdir(DdePair.SAVE_DIR + "/" + str(self.size)):
			os.mkdir(DdePair.SAVE_DIR + "/" + str(self.size))
		filename = DdePair.default_save_name(self.cols, self.size)
		with open(filename, "w") as outfile:
			pickle.dump(self, outfile)

	def _plot_heatmaps(self):
		plt.clf()
		fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
		# fig.set_size_inches(12, 8, forward=True)

		all_min = min(min(self.dde_s.simplex_densities), min(self.dde_b.simplex_densities))
		all_max = max(max(self.dde_s.simplex_densities), max(self.dde_b.simplex_densities))
		norm = mpl.colors.Normalize(vmin=all_min, vmax=all_max)

		points_s = self._plot_heatmap(self.dde_s, fig, ax1, norm)
		points_b = self._plot_heatmap(self.dde_b, fig, ax2, norm)

		ax1.set_ylabel(self.dde_s.columns[1])
		ax1.set_title("signal (%d pts)" % self.dde_s.dt.points.shape[0])
		ax2.set_title("background (%d pts)" % self.dde_b.dt.points.shape[0])

		# print points_s[:,:,0].flatten()[:10]
		# exit()
		xvals = np.concatenate((
			points_s[:,:,0].flatten(),
			points_b[:,:,0].flatten()
		))
		yvals = np.concatenate((
			points_s[:,:,1].flatten(),
			points_b[:,:,1].flatten()
		))
		xmin = min(xvals)
		xmax = max(xvals)
		ymin = min(yvals)
		ymax = max(yvals)
		ax1.set_xlim([xmin, xmax])
		ax1.set_ylim([ymin, ymax])

	def _plot_heatmap(self, dde, fig, ax, norm):
		data = (dde.dt.points[dde.dt.simplices] * dde.stddevs[None,None,:]) + dde.means[None,None,:]
		coll = PolyCollection(
			data,
			array=dde.simplex_densities,
			cmap=mpl.cm.jet,
			norm=norm,
			edgecolors="none"
		)
		ax.add_collection(coll)
		fig.colorbar(coll, ax=ax)
		ax.set_xlabel(dde.columns[0])
		# ax.set_ylabel(dde.columns[1])
		return data

	def save_img(self):
		if not os.path.isdir(DdePair.IMG_SAVE_DIR):
			os.mkdir(DdePair.IMG_SAVE_DIR)
		if not os.path.isdir(DdePair.IMG_SAVE_DIR + "/" + str(self.size)):
			os.mkdir(DdePair.IMG_SAVE_DIR + "/" + str(self.size))
		self._plot_heatmaps()
		filename = DdePair.default_img_name(self.cols, self.size)
		plt.savefig(filename)
		plt.close()

	def show_img(self):
		plt.ion()
		self._plot_heatmaps()
		plt.show()
		_ = raw_input("Enter to continue...")


print "TRAIN_LIMIT:", TRAIN_LIMIT
print "TEST_LIMIT:", TEST_LIMIT
print "USE_MULTIPROCESSING:", USE_MULTIPROCESSING
print "USE_MANUAL_TESTS:", USE_MANUAL_TESTS
print

write("loading training data")
traindata = loadTrainingData(TRAIN_LIMIT * 5) # we'll filter down to TRAIN_LIMIT later, for now oversample
feature_cols = featureCols()
writeDone()

# print "initializing all Delaunay density estimators..."
def init_dde(colpair):
	global TRAIN_LIMIT, traindata
	(from_disk, dde_pair) = DdePair.get_pair(colpair, TRAIN_LIMIT, traindata)
	if not from_disk:
		dde_pair.save()
	# dde_pair.save_img()
	if from_disk:
		print "loaded from disk: %s" % str(colpair)
	else:
		print "calced from scratch: %s" % str(colpair)
	return dde_pair

# col_pairs = MANUAL_TESTS if USE_MANUAL_TESTS else list(pairs(feature_cols))
# if USE_MULTIPROCESSING:
# 	pool = multiprocessing.Pool()
# 	dde_pairs = pool.map(init_dde, col_pairs)
# else:
# 	dde_pairs = map(init_dde, col_pairs)
# # dde_lookup = dict(zip(col_pairs, dde_pairs))

write("Loading DDE")
dde = init_dde(["PRI_lep_phi", "PRI_met_phi"])
writeDone()

write("Loading test data")
testdata = loadTestData(TEST_LIMIT)
writeDone()

write("Scoring test data")
testdata["Scores"] = dde.score(testdata[["PRI_lep_phi", "PRI_met_phi"]])
testdata["Class"] = ["s" if score > 0.0 else "b" for score in testdata["Scores"]]
writeDone()

write("Formatting results")
testdata = testdata.sort("Scores")
testdata["RankOrder"] = np.array(range(1, testdata.shape[0]+1))
testdata = testdata.sort("EventId")
testdata[["EventId", "RankOrder", "Class"]].to_csv("binnedBayesND.csv", header=True, index=False)
writeDone()

global_elapsed = time.time() - global_start
print "Took %s" % fmtTime(global_elapsed)

# saveDensityHeadmap(dde)