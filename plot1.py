import numpy as np
import pylab as P
import sqlite3, linq, signal, sys

conn = sqlite3.connect("data.sqlite")
cursor = conn.cursor()

colInfos = cursor.execute("PRAGMA table_info(training)").fetchall()
cols = [row[1] for row in colInfos]

def getData(col):
	sql = "SELECT %s FROM training WHERE %s != -999.0" % (col, col)
	data = cursor.execute(sql).fetchall()
	data = [row[0] for row in data]
	data = np.array(data)
	return (type(data[0]).__name__, data)

def hist(col):
	
	(typename, data) = getData(col)

	if typename != "float64":
		print "skipping '%s' (type=%s)" % (col, typename)
		return

	sys.stdout.write("showing '%s'..." % col)
	P.clf()
	P.hist(data, 200)
	P.title(col)
	_ = raw_input()

P.ion()
for col in cols:
	hist(col)

# print getData(cols[0])
# print set(type(x).__name__ for x in getData(cols[0]))