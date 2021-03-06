import unittest
import numpy as np

from bspkde import *
from numpy.testing import *

class TestFunctions(unittest.TestCase):
	def test_zscore(self):
		col = np.array([1.0, 2.0, 3.0])
		zscores = z_score(col)
		sd = np.std(col)
		assert_array_equal(zscores, np.array([-1.0/sd, 0.0, 1.0/sd]))

	def test_zscores(self):
		df = pd.DataFrame([
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0]
		])
		sd1 = np.std([1.0, 4.0, 7.0])
		sd2 = np.std([2.0, 5.0, 8.0])
		sd3 = np.std([3.0, 6.0, 9.0])
		expected = pd.DataFrame([
			[-3.0/sd1, -3.0/sd2, -3.0/sd3],
			[0.0/sd1, 0.0/sd2, 0.0/sd3],
			[3.0/sd1, 3.0/sd2, 3.0/sd3]
		])
		assert_array_equal(z_scores(df), expected)

	def test_truth_combinations(self):
		expected = [
			(False, False, False),
			(False, False, True),
			(False, True, False),
			(False, True, True),
			(True, False, False),
			(True, False, True),
			(True, True, False),
			(True, True, True)
		]
		expected_count = len(expected) # 8

		count = 0
		for index, com in enumerate(truth_combinations(3)):
			count += 1
			self.assertEqual(com, expected[index])
		self.assertEqual(count, expected_count)

class TestPartition(unittest.TestCase):
	def setUp(self):
		self.pts = np.array([
			[0.0, 0.0],

			[0.0, 1.0],
			[1.0, 1.0],
			[1.0, 0.0],
			[1.0, -1.0],
			[0.0, -1.0],
			[-1.0, -1.0],
			[-1.0, 0.0],
			[-1.0, 1.0],

			[0.0, 2.0],
			[2.0, 2.0],
			[2.0, 0.0],
			[2.0, -2.0],
			[0.0, -2.0],
			[-2.0, -2.0],
			[-2.0, 0.0],
			[-2.0, 2.0]
		])

		self.min_outer = np.array([-2.0, -2.0])
		self.max_outer = np.array([2.0, 2.0])

		self.min_inner = np.array([-1.0, -1.0])
		self.max_inner = np.array([1.0, 1.0])

	def test_volume(self):
		p = Partition("foo", self.pts, self.min_outer, self.max_outer)
		self.assertEqual(16.0, p.volume())

	def test_density(self):
		p = Partition("foo", self.pts, self.min_outer, self.max_outer)
		self.assertEqual(
			p.density,
			1.0 / 16.0
		)

	def test_is_in_partition(self):
		expected = np.array(
			[True] + # origin
			([True] * 8) + # inner square
			([False] * 8) # outer square
		)

		p = Partition("foo", self.pts, self.min_inner, self.max_inner)
		in_partition = p.is_in_partition(self.pts)

		assert_array_equal(expected, in_partition)

	def test_filter(self):
		expected = self.pts[:9] # origin + inner square
		p = Partition("foo", self.pts, self.min_inner, self.max_inner)
		filtered = p.filter(self.pts)
		assert_array_equal(expected, filtered)

	def test_filter_edge_cases(self):
		expected = np.array([
			True,	# [0.0, 0.0],

			False, # [0.0, 1.0],
			False, # [1.0, 1.0],
			True, # [1.0, 0.0],
			True, # [1.0, -1.0],
			True, # [0.0, -1.0],
			True, # [-1.0, -1.0],
			True, # [-1.0, 0.0],
			False, # [-1.0, 1.0],

			False, # [0.0, 2.0],
			False, # [2.0, 2.0],
			False, # [2.0, 0.0],
			False, # [2.0, -2.0],
			False, # [0.0, -2.0],
			False, # [-2.0, -2.0],
			False, # [-2.0, 0.0],
			False # [-2.0, 2.0]
		])
		expected = self.pts[expected]

		p = Partition(
			"foo",
			self.pts,
			self.min_inner, self.max_inner,
			include_max=np.array([True, False])
		)
		filtered = p.filter(self.pts)
		assert_array_equal(expected, filtered)

	def test_split(self):
		# 0 - min, min
		# 1 - min, max
		# 2 - max, min
		# 3 - max, max
		expected_sub_partitions = np.array([
			3, # [0.0, 0.0],

			3, # [0.0, 1.0],
			3, # [1.0, 1.0],
			3, # [1.0, 0.0],
			2, # [1.0, -1.0],
			2, # [0.0, -1.0],
			0, # [-1.0, -1.0],
			1, # [-1.0, 0.0],
			1, # [-1.0, 1.0],

			3, # [0.0, 2.0],
			3, # [2.0, 2.0],
			3, # [2.0, 0.0],
			2, # [2.0, -2.0],
			2, # [0.0, -2.0],
			0, # [-2.0, -2.0],
			1, # [-2.0, 0.0],
			1 # [-2.0, 2.0]
		])

		expected_mins = np.array([
			[-2.0, -2.0],
			[-2.0, 0.0],
			[0.0, -2.0],
			[0.0, 0.0]
		])
		expected_maxs = np.array([
			[0.0, 0.0],
			[0.0, 2.0],
			[2.0, 0.0],
			[2.0, 2.0]
		])

		expected_include_maxs = np.array([
			[False, False],
			[False, True],
			[True, False],
			[True, True]
		])

		p = Partition("foo", self.pts, self.min_outer, self.max_outer)
		splits = p.split()

		for i in range(4):
			s = splits[i]
			assert_array_equal(expected_mins[i], s.min_corner)
			assert_array_equal(expected_maxs[i], s.max_corner)
			assert_array_equal(expected_include_maxs[i], s.include_max)
			assert_array_equal(
				self.pts[expected_sub_partitions == i],
				s.points
			)

	def test_get_child_indices(self):
		p = Partition("foo", self.pts, self.min_outer, self.max_outer)
		p.children = p.split() #simulated training

		indices = p.get_child_indices(self.pts)
		expected = np.array([
			3, 3, 3, 3, 2, 2, 0, 1, 1, 3, 3, 3, 2, 2, 0, 1, 1
		])
		assert_array_equal(indices, expected)

		for child_index, child_partition in enumerate(p.children):
			assert_array_equal(
				child_partition.is_in_partition(self.pts),
				indices == child_index
			)

	def test_get_density_estimates(self):
		p = Partition("foo", self.pts, self.min_outer, self.max_outer)
		p.children = p.split() #simulated training

		quadrant_counts = np.array([2, 4, 4, 7]).astype(np.float64)
		quadrant_pcts = quadrant_counts / np.sum(quadrant_counts)
		quadrant_densities = quadrant_pcts / 4.0 # shortcut - all volumes are 4.0
		expected_quadrants = np.array([
			3, 3, 3, 3, 2, 2, 0, 1, 1, 3, 3, 3, 2, 2, 0, 1, 1
		])
		expected_densities = quadrant_densities[expected_quadrants]

		densities = p.get_density_estimates(self.pts)
		assert_array_equal(expected_densities, densities)

if __name__ == "__main__":
	unittest.main()