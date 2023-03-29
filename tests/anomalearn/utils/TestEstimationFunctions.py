import unittest

import numpy as np

from anomalearn.utils import estimate_mean_covariance


class TestEstimationFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = np.arange(100).reshape((-1, 1)).astype(np.double)
        cls.vectors = np.arange(200).reshape((-1, 2)).astype(np.double)
    
    def test_estimate_mean_covariance(self):
        mean, cov, inv_cov = estimate_mean_covariance(self.points)
        np.testing.assert_array_equal(np.ma.mean(self.points, axis=0), mean)
        np.testing.assert_array_equal(np.ma.std(self.points, axis=0, ddof=1), cov)
        np.testing.assert_array_equal(1 / (np.ma.std(self.points, axis=0, ddof=1)), inv_cov)

        mean, cov, inv_cov = estimate_mean_covariance(self.vectors)
        np.testing.assert_array_equal(np.ma.mean(self.vectors, axis=0), mean)
        np.testing.assert_array_equal(np.ma.cov(self.vectors, rowvar=False, ddof=1), cov)
        np.testing.assert_array_equal(np.linalg.inv(np.ma.cov(self.vectors, rowvar=False, ddof=1)), inv_cov)
        
        new_points = self.points.copy()
        new_points[:8, :] = np.nan
        masked_new_points = np.ma.array(new_points, mask=np.isnan(new_points))
        mean, cov, inv_cov = estimate_mean_covariance(new_points)
        np.testing.assert_array_equal(np.ma.mean(masked_new_points, axis=0), mean)
        np.testing.assert_array_equal(np.ma.std(masked_new_points, axis=0, ddof=1), cov)
        np.testing.assert_array_equal(1 / (np.ma.std(masked_new_points, axis=0, ddof=1)), inv_cov)
        
        new_vectors = self.vectors.copy()
        new_vectors[:4, 0] = np.nan
        new_vectors[4:4, 1] = np.nan
        masked_new_vectors = np.ma.array(new_vectors, mask=np.isnan(new_vectors))
        mean, cov, inv_cov = estimate_mean_covariance(new_vectors)
        np.testing.assert_array_equal(np.ma.mean(masked_new_vectors, axis=0), mean)
        np.testing.assert_array_equal(np.ma.cov(masked_new_vectors, rowvar=False, ddof=1), cov)
        np.testing.assert_array_equal(np.linalg.inv(np.ma.cov(masked_new_vectors, rowvar=False, ddof=1)), inv_cov)
