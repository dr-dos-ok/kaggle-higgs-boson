import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3, sys

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
cursor.close()

def getData(col):
	sql = "SELECT %s as data, Label FROM training WHERE %s != -999.0" % (col, col)
	data = pd.read_sql(sql, conn)
	# data.data = np.log(data.data)
	return data

def drawData(col):
	plt.clf()
	data = getData(col)
	s = data[data.Label == "s"]
	b = data[data.Label == "b"]
	sns.distplot(b.data, hist=True)
	sns.distplot(s.data, hist=True, color="red")
	plt.title(col)

plt.ion()
for col in cols:
	sys.stdout.write("%s..." % col)
	sys.stdout.flush()
	drawData(col)
	_ = raw_input(" (Done)")