import pandas as pd
import numpy as np
import sqlite3, sys, time, math, signal

_write_time_stack = []
_last_call_was_write = None
def write(s):
	global _write_time_stack, _last_call_was_write

	prefix = ""
	if len(_write_time_stack) > 0:
		if _last_call_was_write:
			prefix += "\n"
		prefix += "\t" * len(_write_time_stack)

	_write_time_stack.append(time.time())
	sys.stdout.write(prefix + s + "...")
	sys.stdout.flush()

	_last_call_was_write = True

def writeDone(elapsed=None):
	global _last_call_was_write

	fmt = "Done - %s"
	if elapsed == None:
		global _write_time_stack
		elapsed = time.time() - _write_time_stack.pop()
		if _last_call_was_write:
			fmt = " (%s)" % fmt
		else:
			fmt = ("\t" * len(_write_time_stack)) + fmt

	print fmt % fmtTime(elapsed)

	_last_call_was_write = False

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

_SIGINT_CAUGHT = False
def handle_sigint(signal, frame):
	global _SIGINT_CAUGHT
	_SIGINT_CAUGHT = True

def capture_sigint():
	global _SIGINT_CAUGHT
	_SIGINT_CAUGHT = False
	signal.signal(signal.SIGINT, handle_sigint)

def uncapture_sigint():
	global _SIGINT_CAUGHT
	_SIGINT_CAUGHT = False
	signal.signal(signal.SIGINT, signal.SIG_DFL)

def is_cancelled():
	return _SIGINT_CAUGHT

def reset_cancelled():
	global _SIGINT_CAUGHT
	_SIGINT_CAUGHT = False

_conn = None
def dbConn():
	global _conn
	if _conn == None:
		_conn = sqlite3.connect("data.sqlite")
	return _conn

_trainingData = None
_trainingDataLoaded = False
def loadTrainingData(numRows=None, col_flag=None, col_flag_str=None):
	global _trainingData, _trainingDataLoaded
	if not _trainingDataLoaded:
		sql = "SELECT * FROM training"
		if col_flag is not None:
			sql += " WHERE col_flag = " + str(col_flag) #too lazy to use parametrized queries; not worried about sql injection here, lol :)
		if col_flag_str is not None:
			sql += " WHERE col_flag_str = '{0}'".format(col_flag_str)
		if numRows != None:
			sql += " LIMIT %d" % numRows
		_trainingData = pd.read_sql(sql, dbConn())
		_trainingData = _trainingData.applymap(lambda x: np.nan if x == -999.0 else x)
		_trainingData = _trainingData.set_index("EventId")
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

_timers = []

def start_timer(name):
	global _timers
	_timers.append((name, True, time.time()))

def stop_timer(name):
	global _timers
	_timers.append((name, False, time.time()))

def print_timers():
	global _timers
	timer_dict = {}
	timer_starts = {}
	timer_order = []
	for name, is_start, event_time in _timers:
		if name not in timer_dict:
			timer_dict[name] = 0.0
			timer_order.append(name)
		if is_start:
			timer_starts[name] = event_time
		else:
			prev_total = timer_dict[name]
			new_total = prev_total + (event_time - timer_starts[name])
			timer_starts[name] = None
			timer_dict[name] = new_total

	print
	total_seconds = 0.0
	for name in timer_order:
		seconds = timer_dict[name]
		total_seconds += seconds
		print "%s:	%s" % (name, fmtTime(seconds))
	print "TOTAL:	%s" % fmtTime(total_seconds)

_TOTAL_S = 691.0
_TOTAL_B = 410000.0
def ams(predictions, actual_df):

	actual_df = actual_df.copy() # copy this so we don't mess up someone else's data

	predicted_signal = predictions == "s"
	actual_signal = actual_df["Label"] == "s"
	actual_background = ~actual_signal

	total_s = np.sum(actual_df[actual_signal]["Weight"])
	total_b = np.sum(actual_df[actual_background]["Weight"])

	actual_df.loc[actual_signal, "Weight"] *= _TOTAL_S / total_s
	actual_df.loc[actual_background, "Weight"] *= _TOTAL_B / total_b

	s = true_positives = np.sum(actual_df[predicted_signal & actual_signal]["Weight"])
	b = false_positives = np.sum(actual_df[predicted_signal & actual_background]["Weight"])
	B_R = 10.0

	radicand = 2.0 * ((s+b+B_R) * math.log(1.0 + (s/(b+B_R))) - s)

	if radicand < 0.0:
		print "radicand is less than 0, exiting"
		exit()
	else:
		return math.sqrt(radicand)

if __name__ == "__main__":
	print fmtTime(1.23)
	print fmtTime(66.1)
	print fmtTime(120)
	print fmtTime(4*60*60 + 5*60 + 10)