import sqlite3, sys

def write(s):
	sys.stdout.write(s+"...")
	sys.stdout.flush()

def writeDone():
	print " (Done)"

def dbConn():
	return sqlite3.connect("data.sqlite")

