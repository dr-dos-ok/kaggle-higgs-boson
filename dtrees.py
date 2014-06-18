import pandas as pd
import numpy as np
import time, bkputils, math

class BinStump:
	def __init__(self, data):
		sortedData = np.sort(npData)
		numBins = int(math.sqrt(data.size))
		bindices = np.linspace(0, data.size-1, numBins+1).astype(np.int32)
		bins = sortedData[bindices]