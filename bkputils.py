import pandas as pd
import numpy as np
import sqlite3, sys, time, math

_write_time = None

def write(s):
	global _write_time
	_write_time = time.time()
	sys.stdout.write(s+"...")
	sys.stdout.flush()

def writeDone(elapsed=None):
	fmt = "Done - %s"
	if elapsed == None:
		global _write_time
		elapsed = time.time() - _write_time
		fmt = " (%s)" % fmt
	print fmt % fmtTime(elapsed)

def fmtTime(secs):

	hrs = math.floor(secs / (60.**2))
	secs = secs - (hrs * (60**2))
	mins = math.floor(secs / 60.)
	secs = secs - (mins * 60)

	if hrs > 0:
		return "%d:%02d:%02d hrs" % (hrs, mins, int(secs))
	elif mins > 0:
		return "%d:%02d mins" % (mins, int(secs))
	else:
		return "%.4f secs" % secs

_conn = None
def dbConn():
	global _conn
	if _conn == None:
		_conn = sqlite3.connect("data.sqlite")
	return _conn

_trainingData = None
_trainingDataLoaded = False
def loadTrainingData(numRows=None):
	global _trainingData, _trainingDataLoaded
	if not _trainingDataLoaded:
		sql = "SELECT * FROM training"
		if numRows != None:
			sql += " LIMIT %d" % numRows
		_trainingData = pd.read_sql(sql, dbConn())
		_trainingData = _trainingData.applymap(lambda x: np.nan if x == -999.0 else x)
		_trainingDataLoaded = True
	return _trainingData

_featureCols = None
def featureCols(only_float64=True):
	global _featureCols
	if _featureCols == None:
		traindata = loadTrainingData()
		_featureCols = [
			colName
			for colName in traindata.columns.values if colName.startswith("DER") or colName.startswith("PRI")
		]
		if only_float64:
			_featureCols = filter(lambda colName: _trainingData.dtypes[colName] == np.float64, _featureCols)
	return _featureCols

_testData = None
_testDataLoaded = False
def loadTestData(numRows=None):
	global _testData, _testDataLoaded
	if not _testDataLoaded:
		sql = sql = "SELECT EventId, %s FROM test" % (", ".join(featureCols()))
		if numRows != None:
			sql += " LIMIT %d" % numRows
		_testData = pd.read_sql(sql, dbConn())
		_testData = _testData.applymap(lambda x: np.nan if x == -999.0 else x)
		_testDataLoaded = True
	return _testData

if __name__ == "__main__":
	print fmtTime(1.23)
	print fmtTime(66.1)
	print fmtTime(120)
	print fmtTime(4*60*60 + 5*60 + 10)