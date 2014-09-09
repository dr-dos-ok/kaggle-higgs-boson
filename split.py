"""
Split the training set into training and validation sets
randomly. Then save off the EventIds of each set so we can
reuse them later
"""

import numpy as np
import pandas as pd

from bkputils import *

VALIDATION_PCT = 0.2

_FILENAME = "split_ids.npz"

def load_ids():
	with open(_FILENAME, "r") as infile:
		npzfile = np.load(infile)
		validation_ids = npzfile["validation_ids"]
		train_ids = npzfile["train_ids"]
		return validation_ids, train_ids

def save_ids():
	write("loading train data")
	traindata = loadTrainingData()
	writeDone()

	write("splitting EventIds")
	all_ids = traindata["EventId"].values
	num_validation = int(np.round(traindata.shape[0] * VALIDATION_PCT))
	validation_ids = np.random.choice(all_ids, size=num_validation, replace=False)
	validation_ids = np.sort(validation_ids)
	train_ids = np.setdiff1d(all_ids, validation_ids)
	writeDone()

	write("saving EventIds")
	with open(_FILENAME, "w") as outfile:
		np.savez(outfile, validation_ids=validation_ids, train_ids=train_ids)
	writeDone()

if __name__ == "__main__":
	save_ids()

	# validation_ids, train_ids = load_ids()
	# print len(validation_ids), validation_ids[:10]
	# print len(train_ids), train_ids[:10]