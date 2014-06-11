import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
import sqlite3, sys, time, multiprocessing

global_start = time.time()

def write(s):
	sys.stdout.write(s+"...")
	sys.stdout.flush()

def writeDone():
	print " (Done)"

def makeKde(coldata):
	coldata = coldata.reshape(-1, 1)
	colmin = min(coldata)
	colmax = max(coldata)
	colrange = np.abs(colmax - colmin)
	bandwidth = colrange / 100.0

	return KernelDensity(bandwidth=bandwidth, kernel = 'tophat').fit(coldata)

conn = sqlite3.connect("data.sqlite")

write("loading sql metadata")
cursor = conn.cursor()
colInfos = cursor.execute("PRAGMA table_info(training)").fetchall()
cols = [row[1] for row in colInfos]
cursor.close()
writeDone()

write("loading training data")
train_s = pd.read_sql("SELECT * FROM training WHERE Label='s'", conn)
train_b = pd.read_sql("SELECT * FROM training WHERE Label='b'", conn)

resample_size = 1000

numrows = train_s.shape[0]
train_s = train_s.loc[np.random.choice(range(0, numrows), resample_size)]
train_s.set_index([range(0, resample_size)], inplace=True)

numrows = train_b.shape[0]
train_b = train_b.iloc[np.random.choice(range(0, numrows), resample_size)]
train_b.set_index([range(0, resample_size)], inplace=True)

colTypes = {}
for col in cols:
	colTypes[col] = type(train_s.at[0,col])
floatCols = [col for col in cols if colTypes[col] == np.float64]
floatCols.remove("Weight")
writeDone()

write("creating kdes")
s_kdes = {}
b_kdes = {}
for col in floatCols:
	coldata_s = train_s[train_s[col] != -999.0][col]
	coldata_b = train_b[train_b[col] != -999.0][col]
	
	s_kdes[col] = makeKde(coldata_s)
	b_kdes[col] = makeKde(coldata_b)
writeDone()

write("loading test data")
sql = "SELECT EventId, %s FROM test" % ", ".join(floatCols)
test = pd.read_sql(sql, conn)
numrows = test.shape[0]
writeDone()

print "making predictions (this will take a while)..."
spredictions = []

rowcounter = multiprocessing.Value('i', 0)
def calcPrediction(rowtuple):
	global rowcounter
	rowcounter.value += 1
	(rowindex, row) = rowtuple

	if rowcounter.value % 1000 == 0:
		sys.stdout.write("\r%d of %d (%f%%)      " % (rowcounter.value, numrows, rowcounter.value*100.0/numrows))
		sys.stdout.flush()
	bprobs = np.exp(np.array([b_kdes[col].score(row[col]) for col in floatCols if row[col] != -999.0]))
	sprobs = np.exp(np.array([s_kdes[col].score(row[col]) for col in floatCols if row[col] != -999.0]))
	sumprobs = bprobs + sprobs

	return np.mean(sprobs / sumprobs)

spredictions = multiprocessing.Pool(8).map(calcPrediction, test.iterrows())
print "\rDone.                 "

write("formatting and writing results")
test["spredictions"] = spredictions
test["Class"] = ["s" if spred >= 0.5 else "b" for spred in spredictions]

submission = test[["EventId", "Class", "spredictions"]].sort("spredictions")
submission["RankOrder"] = range(1, len(spredictions)+1)
submission = submission.sort("EventId")
# print submission.head()
submission[["EventId", "RankOrder", "Class"]].to_csv("kdeClassifier.csv", header=True, index=False)
writeDone()

global_elapsed = time.time() - global_start
print "Took %d:%d" % (global_elapsed/60, global_elapsed)