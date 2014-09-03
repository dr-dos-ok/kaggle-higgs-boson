import numpy as np

# not really general purpose - intended specifically for
# kaggle higgs boson challenge

class ZNetClassifier(object):
	def __init__(self, znet):
		self.znet = znet

	def classify(self, dataframe):
		score = self.znet.forward(dataframe)
		options = np.array(["b", "s"])
		classes = options[(score > 0.0).astype(np.int).ravel()]
		confidence = score ** 2
		return (classes, confidence)