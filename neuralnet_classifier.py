import numpy as np

# not really general purpose - intended specifically for
# kaggle higgs boson challenge

class ZNetClassifier(object):
	def __init__(self, znet):
		self.znet = znet
		self.cutoff = 0.0

	def classify(self, dataframe):
		score = self.znet.forward(dataframe)
		options = np.array(["b", "s"])
		classes = options[(score[:,0] - score[:,1] > self.cutoff).astype(np.int).ravel()]
		confidence = np.max(score, axis=1) ** 2
		return (classes, confidence)