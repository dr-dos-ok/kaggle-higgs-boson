import unittest
import numpy as np
from voronoi import *
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

	def test_calc_simplex_volumes(self):
		simplices = np.array(
			[
				[
					[0.0,0.0,0.0],
					[2.0, 4.0, 0.0],
					[5.0, 0.0, 0.0],
					[3.0, 1.0, 4.0]
				]
			],
			np.float64
		)
		expected = np.array([40.0/3.0])
		actual = calc_simplex_volumes(simplices=simplices)
		assert_array_almost_equal_nulp(actual, expected, nulp=3)

		simplices = np.array(
			[
				[
					[0.0,0.0],
					[2.0, 4.0],
					[5.0, 0.0]
				]
			],
			np.float64
		)
		expected = np.array([10.0])
		actual = calc_simplex_volumes(simplices=simplices)
		assert_array_almost_equal_nulp(actual, expected, nulp=1)

class TestVoronoiKde(unittest.TestCase):

	def setUp(self):
		self.kde = VoronoiKde(
			"test_kde",
			pd.DataFrame([
				#bin points
				[0.0, 0.0],		# 0
				[0.0, 4.0],		# 1
				[6.0, 0.0],		# 2
				[0.0, -8.0],	# 3
				[-2.0, 0.0],	# 4

				#fill points
				[-2.0, -1.0],	# 4
				[0.0, 3.0],		# 1
				[1.0, 0.0],		# 0
				[2.0, 0.0],		# 0
				[2.0, -2.0],	# 0
				[4.0, 0.0],		# 2
				[5.0, 0.0]		# 2
			]),
			[0, 1],
			bin_indices=range(5)
		)

		expected_volumes = np.array([
			24.0,	# 0
			4.0,	# 1
			9.0,	# 2
			8.0,	# 3
			3.0		# 4
		])
		expected_counts = np.array([
			4,	# 0
			2,	# 1
			3,	# 2
			1,	# 3
			2	# 4
		])
		expected_densities = (expected_counts / expected_volumes) / np.sum(expected_counts)

		self.expected_counts = expected_counts
		self.expected_volumes = expected_volumes
		self.expected_densities = expected_densities

	# def test_expectations(self):
	# 	assert_equal(np.sum(self.expected_densities * self.expected_volumes), 1.0)

	def test_kde(self):

		print self.kde.bin_densities
		print self.expected_densities

		assert_array_almost_equal_nulp(
			self.kde.bin_densities,
			self.expected_densities,
			nulp=3
		)


if __name__ == "__main__":
	unittest.main()
	# print z_score(np.array([1.0, 4.0, 7.0]))