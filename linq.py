from itertools import *

try:
	import numpy as np
	_got_numpy = True
except ImportError:
	_got_numpy = False

_truefn = lambda x: True
_falsefn = lambda y: False
_identity = lambda x: x

class Linq:

	def __init__(self, iterator):
		self._iterator = iterator

	def all(self, fn=_identity):
		for item in self._iterator:
			if not fn(item):
				return False
		return True

	def any(self, fn=_truefn):
		for item in self._iterator:
			if fn(item):
				return True
		return false

	def distinct(self, fn):
		result = set()
		for item in this._iterator:
			if item not in result:
				result.add(item)
		return Linq(result)

	def first(self, fn=None):
		(found, result) = self.tryFirst(fn)
		if found:
			return result
		else:
			raise Exception("Linq.first() failed to find a matching item")

	def firstOrNone(self, fn=None):
		(_, result) = self.tryFirst(fn)
		return result

	def groupBy(self, fn):
		groups = []
		groupDict = {}
		for item in self._iterator:
			key = fn(item)
			if key in groupDict:
				group = groupDict[key]
			else:
				group = Group(key)
				groups.append(group)
				groupDict[key] = group
			group.group.append(item)
		return Linq(groups)

	def last(self, fn=None):
		(found, result) = self.tryLast(fn)
		if found:
			return result
		else:
			raise Exception("Linq.last() failed to find a matching item")

	def lastOrNone(self, fn=None):
		(_, result) = self.tryLast(fn)
		return result

	def reversed(self):
		return Linq(reversed(list(self._iterator)))

	def select(self, fn=_identity):
		def generator():
			for item in self._iterator:
				yield fn(item)
		return Linq(generator())

	def selectMany(self, fn=_identity):
		def generator():
			for item in self._iterator:
				for subItem in fn(item):
					yield subItem
		return Linq(generator())

	def sort(self, fn=None, asc=True):
		if fn == None:
			l = sorted(self._iterator)
		else:
			l = sorted(fn(item) for item in self._iterator)

		if not asc:
			l = reversed(l)

		return Linq(l)

	def toDict(self, keyFn, valFn=_identity):
		result = {}
		for item in self._iterator:
			key = keyFn(item)
			if key in result:
				raise Exception("key returned twice in Linq.toDict(): '%s'", key)
			result[key] = valFn(item)
		return result

	def toList(self):
		return [item for item in self._iterator]

	def toNumpyArray(self, numpyType):
		if not _got_numpy:
			raise Exception("It seems numpy did not import successfully. You'll have to install it before you can call toNumpyArray successfully")
		return np.fromiter(self._iterator, numpyType)

	def tryFirst(self, fn=_truefn):
		for item in self._iterator:
			if fn(item):
				return (True, item)
		return (False, None)

	def tryLast(self, fn=None):
		return self.reversed().tryFirst(fn)

	def where(self, fn):
		def generator():
			for item in self._iterator:
				if fn(item):
					yield item
		return Linq(generator())

class Group:
	def __init__(self, key):
		self.key = key
		self.group = []

def wrap(iterable):
	return Linq(iterable)

def range(a, b=None, incr=None):
	if b==None:
		start = 0
		end = a
	else:
		start = a
		end = b
	
	if start == end:
		gen = _emptyGenerator()
	elif start < end:
		incr = 1 if incr == None else incr
		gen = _rangeAsc(start, stop, incr)
	else:
		incr = -1 if incr == None else incr
		gen = _rangeDesc(start, stop, incr)
	return Linq(gen)

def _emptyGenerator():
	return
	yield

def _rangeAsc(start, stop, incr):
	index = start
	while index < end:
		yield index
		index += incr

def _rangeDesc(start, stop, incr):
	index = start
	while index > end:
		yield index
		index += incr

