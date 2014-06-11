import numpy as np
import pylab as P
import sqlite3, linq, signal, sys
from matplotlib.ticker import NullFormatter
from sklearn.neighbors.kde import KernelDensity

conn = sqlite3.connect("data.sqlite")
cursor = conn.cursor()

colInfos = cursor.execute("PRAGMA table_info(training)").fetchall()
cols = [row[1] for row in colInfos]

colTypes = {}
for col in cols:
	sql = "SELECT %s FROM training WHERE %s != -999.0 LIMIT 1" % (col, col)
	data = cursor.execute(sql).fetchall()
	colTypes[col] = type(data[0][0])
cols = [col for col in cols if colTypes[col] == float]

def getData(col1, col2):
	sql = sql = "SELECT %s, %s, Label FROM training WHERE %s != -999.0 AND %s != -999.0" % (col1, col2, col1, col2)
	data = cursor.execute(sql).fetchall()
	type1 = type(data[0][0]).__name__
	type2 = type(data[0][1]).__name__
	return ((type1, type2), data)

def getPairs(l1, l2):
	for i1 in range(0, len(l1)):
		for i2 in range(i1+1, len(l2)):
			yield (l1[i1], l2[i2])

def doplot(col1, col2):
	(types, data) = getData(col1, col2)

	scatterSize, margin = 0.7, 0.07
	histSize = 1.0 - scatterSize - (margin*3)
	scatterBox = [margin, margin, scatterSize, scatterSize]
	topHistBox = [margin, margin+scatterSize+margin, scatterSize, histSize]
	rightHistBox = [margin+scatterSize+margin, margin, histSize, scatterSize]

	P.clf()
	P.figure(1, figsize=(8,8))

	scatterAxes = P.axes(scatterBox)
	topHistAxes = P.axes(topHistBox)
	rightHistAxes = P.axes(rightHistBox)

	nullFormatter = NullFormatter()
	topHistAxes.yaxis.set_major_formatter(nullFormatter)
	rightHistAxes.xaxis.set_major_formatter(nullFormatter)

	s = [row for row in data if row[2] == 's']
	b = [row for row in data if row[2] == 'b']

	bx = np.array([row[0] for row in b])
	by = np.array([row[1] for row in b])
	sx = np.array([row[0] for row in s])
	sy = np.array([row[1] for row in s])

	_min = lambda x1, x2: min([min(x1), min(x2)])
	_max = lambda x1, x2: max([max(x1), max(x2)])
	xmin = _min(bx, sx)
	xmax = _max(bx, sx)
	ymin = _min(by, sy)
	ymax = _max(by, sy)
	
	xlims = (xmin, xmax)
	ylims = (ymin, ymax)

	def pad(lims):
		padAmt = (lims[1] - lims[0]) * 0.05
		return (lims[0] - padAmt, lims[1] + padAmt)

	xlims = pad(xlims)
	ylims= pad(ylims)

	scatterAxes.set_xlim(xlims)
	scatterAxes.set_ylim(ylims)
	topHistAxes.set_xlim(scatterAxes.get_xlim())
	rightHistAxes.set_ylim(scatterAxes.get_ylim())

	scatterAxes.scatter(bx, by, color="blue", alpha=0.01)
	histLine(topHistAxes, bx, xlims, "blue")
	# topHistAxes.hist(bx, bins=200, orientation="vertical", color="blue")
	# rightHistAxes.hist(by, bins=200, orientation="horizontal", color="blue")

	# scatterAxes.scatter(sx, sy, color="red")
	# topHistAxes.hist(sx, orientation="horizontal", color="red")
	# rightHistAxes.hist(sx, orientation="vertical", color="red")

	# P.title("%s vs %s" % (col1, col2))
	# P.show()

def histLine(axes, data, minmax, color):
	(xmin, xmax) = minmax
	data = data.reshape(-1, 1)
	kde = KernelDensity(bandwidth=(xmax-xmin)/100.0).fit(data)
	x = np.linspace(xmin, xmax, 100).reshape(-1, 1)
	foo = kde.score_samples(x)
	density = np.exp(foo)

	axes.plot(x, density, color=color)

P.ion()
for colPair in getPairs(cols, cols):
	sys.stdout.write("%s vs %s..." % (colPair[0], colPair[1]))
	sys.stdout.flush()
	doplot(colPair[0], colPair[1])
	_ = raw_input(" (Done)")




