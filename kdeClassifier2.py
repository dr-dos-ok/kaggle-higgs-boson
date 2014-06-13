import numpy as np
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

def subsample(df, n):
	(numrows, numcols) = df.shape
	result = df.loc[np.random.choice(xrange(0, numrows), n)]
	result.set_index([range(0, n)], inplace=True)
	return result

conn = sqlite3.connect("data.sqlite")

write("loading sql metadata")
cursor = conn.cursor()
colInfos = cursor.execute("PRAGMA table_info(training)").fetchall()
cols = [row[1] for row in colInfos]
cursor.close()
writeDone()

write("loading training data")
training_s = pd.read_sql("SELECT * FROM training WHERE Label='s'", conn)
training_b = pd.read_sql("SELECT * FROM training WHERE Label='b'", conn)
writeDone()

write("munging training data")
subsample_size = 1000
training_s = subsample(training_s, subsample_size)
training_b = subsample(training_b, subsample_size)

colTypes = {}
for col in cols:
	colTypes[col] = type(training_s.at[0,col])
floatCols = [col for col in cols if colTypes[col] == np.float64]
floatCols.remove("Weight")
writeDone()

write("creating kdes")
kdes_s = {}
kdes_b = {}
for col in floatCols:
	coldata_s = training_s[training_s[col] != -999.0][col]
	coldata_b = training_b[training_b[col] != -999.0][col]

	try:
		kdes_s[col] = makeKde(coldata_s)
	except:
		print coldata_s
		exit()

	kdes_b[col] = makeKde(coldata_b)
training_s = None # may save us some startup time on multiprocessing startup later
training_b = None # may save us some startup time on multiprocessing startup later
writeDone()

numrows = 550000

write("loading test data")
test = pd.read_csv("test.csv")
writeDone()

print "making predictions (this will take a while)..."
spredictions = []
pool = multiprocessing.Pool()


def calcScores(col):
	kde

# def getColProbs(row, kde_dict):
# 	scores = []
# 	for col in floatCols:
# 		val = row[col]
# 		if val == -999.0:
# 			continue
# 		kde = kde_dict[col]
# 		scores.append(np.exp(kde.score(val)))
# 	return np.array(scores)

# rowcounter = multiprocessing.Value("i", 0)
# def calcPrediction(rowtuple):
# 	global rowcounter
# 	rowcounter.value += 1

# 	(rowindex, row) = rowtuple

# 	if rowcounter.value % 1000 == 0:
# 		sys.stdout.write("\r%d of %d (%f%%)      " % (rowcounter.value, numrows, rowcounter.value*100.0/numrows))
# 		sys.stdout.flush()
# 	probs_s = getColProbs(row, kdes_s)
# 	probs_b = getColProbs(row, kdes_b)

# 	return np.mean((probs_s - probs_b)/(probs_s + probs_b))

# threadpool = multiprocessing.Pool()
# test = pd.read_csv("test.csv")
# spredictions = multiprocessing.Pool().map(calcPrediction, test.iterrows())
# print "\rDone.                 "

write("formatting and writing results")
test["spredictions"] = spredictions
test["Class"] = ["s" if spred >= 0.0 else "b" for spred in spredictions]

submission = test[["EventId", "Class", "spredictions"]].sort("spredictions")
submission["RankOrder"] = range(1, len(spredictions)+1)
submission = submission.sort("EventId")
print submission[["EventId", "RankOrder", "Class"]].head()
# submission[["EventId", "RankOrder", "Class"]].to_csv("kdeClassifier2.csv", header=True, index=False)
writeDone()

global_elapsed = time.time() - global_start
print "Took %d:%d" % (global_elapsed/60, global_elapsed%60)



























