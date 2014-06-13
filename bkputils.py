import pandas as pd
import numpy as np
import sqlite3, sys

def write(s):
	sys.stdout.write(s+"...")
	sys.stdout.flush()

def writeDone():
	print " (Done)"

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
def featureCols():
	global _featureCols
	if _featureCols == None:
		traindata = loadTrainingData()
		_featureCols = [colName for colName in list(traindata.columns.values) if _trainingData.dtypes[colName] == np.float64]
		_featureCols.remove("Weight")
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