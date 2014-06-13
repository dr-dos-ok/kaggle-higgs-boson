import numpy as np
import pandas as pd
from bkputils import *
import math, time, sys

global_start = time.time()

class BinSet2D:
	def __init__(self, dataframe):
		xdata = dataframe.ix[:0]
		ydata = dataframe.ix[:1]
		weights = dataframe.ix[:2]

		self._numBins = int(math.sqrt(math.sqrt(xdata.size)))
		xbins = self._customBins(xdata)
		ybins = self._customBins(ydata)

		(self._hist, self._xbins, self._ybins) = np.histogram2d(
			xdata,
			ydata,
			bins = [xbins, ybins],
			normed=True,
			weights = weights
		)

	def _customBins(data):
		sortedData = np.sort(data)
		bindices = np.linspace(0, data.size-1, self._numBins+1).astype(np.int32)
		bins= sortedData[bindices]
		return bins

	def score(self, dataframe):
		xdata = dataframe.ix[:0]
		ydata = dataframe.ix[:1]

		xbindices = np.digitize(xdata, self._xbins) - 1
		ybindices = np.digitize(xdata, self._ybins) - 1

		numBins = self._numBins
		return np.array([
			self._hist[coords] \
			if (
				coords[0] >= 0 and
				coords[1] >= 0 and
				coords[0] < numBins and
				coords[1] < numBins
			) \
			else np.na \
			for coords in zip(xbindices, ybindices)
		])

	def _inRange(point, lowerPoint, upperPoint):
		for i in range(0, len(point)):
			if point[i] < lowerPoint[i] or point[i] >= upperPoint[i]:
				return False
