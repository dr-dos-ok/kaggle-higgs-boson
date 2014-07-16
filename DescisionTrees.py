class DecisionTree:
	def __init__(self, dataframe, remaining_cols, labelcol):
		self.dataframe = dataframe
		self.remaining_cols = remaining_cols
		self.labelcol = labelcol
		self.haschildren = False

	def grow(self):
		if not self.haschildren:
			self.makechildren()
		else:
			maxentropy = 0.0
			maxindex = -1
			for index, child in enumerate(self.children):
				childentropy = child.getentropy()
				if childentropy > maxentropy:
					maxentropy = childentropy
					maxindex = index
			self.children[maxindex].grow()

	def makechildren(self):
		binsets = [
			BinSet(self.dataframe, col)
			for col in self.remaining_cols
		]
		mutualinfos = np.array([
			binset.mutualinfo(self.dataframe, self.labelcol))
			for binset in BinSet
		])
		maxindex = mutualinfos.argmax()
		maxcol = self.remaining_cols[maxindex]
		maxbinset = binsets[maxindex]
		self.childbins = maxbinset

		

	def decide(self, row):
		bindex = self.binset.getbindex(row)
		self.children[bindex].decide(row)

class DecisionLeaf:
	def __init__(self, decision, remaining_entropy):
		self.decision = decision
		self.remaining_entropy = remaining_entropy

	def decide(self, row):
		return self.decision