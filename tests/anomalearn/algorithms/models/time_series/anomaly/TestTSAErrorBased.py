import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal

from tests.anomalearn.algorithms.models.time_series.anomaly.stubs import TSAErrorBasedChild


def my_diff(gt, pred):
    return gt - pred


def my_scores(errors):
    return errors[:, 0]


def my_threshold(errors):
    return np.min(errors[:, 0])


class TestTSASemiSupervised(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gt_points = np.random.rand(100, 1)
        cls.pr_points = np.random.rand(100, 1)
        
        cls.gt_vectors = np.random.rand(100, 3)
        cls.pr_vectors = np.random.rand(100, 3)
    
    def test_compute_errors(self):
        ssm_difference = TSAErrorBasedChild(error_method="difference")
        ssm_abs_difference = TSAErrorBasedChild(error_method="abs_difference")
        ssm_norm = TSAErrorBasedChild(error_method="norm")
        ssm_custom = TSAErrorBasedChild(error_method="custom", error_function=my_diff)
        ssm_wrong = TSAErrorBasedChild()
        
        np.testing.assert_array_equal(ssm_difference._compute_errors(self.gt_points, self.pr_points, False),
                                      self.gt_points - self.pr_points)
        np.testing.assert_array_equal(ssm_difference._compute_errors(self.gt_vectors, self.pr_vectors, False),
                                      self.gt_vectors - self.pr_vectors)
        
        np.testing.assert_array_equal(ssm_abs_difference._compute_errors(self.gt_points, self.pr_points, False),
                                      np.abs(self.gt_points - self.pr_points))
        np.testing.assert_array_equal(ssm_abs_difference._compute_errors(self.gt_vectors, self.pr_vectors, False),
                                      np.abs(self.gt_vectors - self.pr_vectors))
        
        np.testing.assert_array_equal(ssm_norm._compute_errors(self.gt_points, self.pr_points, False),
                                      np.linalg.norm(self.gt_points - self.pr_points, axis=1).reshape((-1, 1)))
        np.testing.assert_array_equal(ssm_norm._compute_errors(self.gt_vectors, self.pr_vectors, False),
                                      np.linalg.norm(self.gt_vectors - self.pr_vectors, axis=1).reshape((-1, 1)))
        
        np.testing.assert_array_equal(ssm_custom._compute_errors(self.gt_points, self.pr_points, False),
                                      my_diff(self.gt_points, self.pr_points))
        np.testing.assert_array_equal(ssm_custom._compute_errors(self.gt_vectors, self.pr_vectors, False),
                                      my_diff(self.gt_vectors, self.pr_vectors))
        
        ssm_wrong.error_method = "zio_paperino_birichino"
        np.testing.assert_array_equal(ssm_wrong._compute_errors(self.gt_points, self.pr_points, False),
                                      np.full(self.gt_points.shape, fill_value=np.nan))
    
    def test_compute_scores(self):
        ssm_gaussian = TSAErrorBasedChild(threshold_computation="gaussian")
        ssm_mahalanobis = TSAErrorBasedChild(threshold_computation="mahalanobis")
        ssm_custom = TSAErrorBasedChild(threshold_computation="custom",
                                        threshold_function=my_threshold,
                                        scoring_function=my_scores)

        point_errors = ssm_custom._compute_errors(self.gt_points, self.pr_points)
        vector_errors = ssm_custom._compute_errors(self.gt_vectors, self.pr_vectors)

        # Check score computation for scalars
        ssm_custom._compute_mean_and_cov(point_errors)
        ssm_gaussian._compute_mean_and_cov(point_errors)
        ssm_mahalanobis._compute_mean_and_cov(point_errors)
        ssm_custom._learn_threshold(point_errors)
        ssm_gaussian._learn_threshold(point_errors)
        ssm_mahalanobis._learn_threshold(point_errors)
        np.testing.assert_array_equal(ssm_custom._compute_scores(point_errors),
                                      my_scores(point_errors))
        self.assertEqual(np.sum(ssm_gaussian._compute_scores(point_errors) -
                                (1 / (multivariate_normal.pdf(point_errors,
                                                              mean=ssm_gaussian._mean,
                                                              cov=ssm_gaussian._cov,
                                                              allow_singular=True) + 1e-10))),
                         0)
        np.testing.assert_array_equal(ssm_mahalanobis._compute_scores(point_errors),
                                      [mahalanobis(point, ssm_mahalanobis._mean, ssm_mahalanobis._inv_cov)
                                       for point in point_errors])

        # Check score computation for vectors
        ssm_custom._compute_mean_and_cov(vector_errors)
        ssm_gaussian._compute_mean_and_cov(vector_errors)
        ssm_mahalanobis._compute_mean_and_cov(vector_errors)
        ssm_custom._learn_threshold(vector_errors)
        ssm_gaussian._learn_threshold(vector_errors)
        ssm_mahalanobis._learn_threshold(vector_errors)
        np.testing.assert_array_equal(ssm_custom._compute_scores(vector_errors),
                                      my_scores(vector_errors))
        self.assertEqual(np.sum(ssm_gaussian._compute_scores(vector_errors) -
                                (1 / (multivariate_normal.pdf(vector_errors,
                                                              mean=ssm_gaussian._mean,
                                                              cov=ssm_gaussian._cov,
                                                              allow_singular=True) + 1e-10))),
                         0)
        np.testing.assert_array_equal(ssm_mahalanobis._compute_scores(vector_errors),
                                      [mahalanobis(point, ssm_mahalanobis._mean, ssm_mahalanobis._inv_cov)
                                       for point in vector_errors])
    
    def test_learn_threshold(self):
        ssm_gaussian = TSAErrorBasedChild(threshold_computation="gaussian")
        ssm_mahalanobis = TSAErrorBasedChild(threshold_computation="mahalanobis")
        ssm_custom = TSAErrorBasedChild(threshold_computation="custom",
                                        threshold_function=my_threshold,
                                        scoring_function=my_scores)

        # Test edge cases for mean and covariance computation
        points_pr = self.gt_points
        vector_pt = self.gt_vectors
        point_errors = ssm_custom._compute_errors(self.gt_points, points_pr)
        vector_errors = ssm_custom._compute_errors(self.gt_vectors, vector_pt)

        ssm_custom._compute_mean_and_cov(point_errors)
        ssm_custom._learn_threshold(point_errors)
        np.testing.assert_array_equal(ssm_custom._mean, np.mean(point_errors, axis=0))
        np.testing.assert_array_equal(ssm_custom._cov, np.std(point_errors, axis=0, ddof=1) + 1e-10)
        np.testing.assert_array_equal(ssm_custom._inv_cov, 1 / (np.std(point_errors, axis=0, ddof=1) + 1e-10))

        ssm_custom._compute_mean_and_cov(vector_errors)
        ssm_custom._learn_threshold(vector_errors)
        np.testing.assert_array_equal(ssm_custom._mean, np.mean(vector_errors, axis=0))
        np.testing.assert_array_equal(ssm_custom._cov, np.cov(vector_errors, rowvar=False, ddof=1) + 1e-10)
        np.testing.assert_array_equal(ssm_custom._inv_cov, np.linalg.pinv(np.cov(vector_errors, rowvar=False, ddof=1) + 1e-10))

        point_errors = ssm_custom._compute_errors(self.gt_points, self.pr_points)
        vector_errors = ssm_custom._compute_errors(self.gt_vectors, self.pr_vectors)

        # Test that the error computation works both for scalar and vectors.
        # Then, test that the mean vector and covariance matrix are correct.
        ssm_custom._compute_mean_and_cov(point_errors)
        ssm_custom._learn_threshold(point_errors)
        np.testing.assert_array_equal(ssm_custom._threshold, np.min(point_errors[:, 0]))
        np.testing.assert_array_equal(ssm_custom._mean, np.mean(point_errors, axis=0))
        np.testing.assert_array_equal(ssm_custom._cov, np.std(point_errors, axis=0, ddof=1))
        np.testing.assert_array_equal(ssm_custom._inv_cov, 1 / np.std(point_errors, axis=0, ddof=1))

        ssm_custom._compute_mean_and_cov(vector_errors)
        ssm_custom._learn_threshold(vector_errors)
        np.testing.assert_array_equal(ssm_custom._threshold, np.min(vector_errors[:, 0]))
        np.testing.assert_array_equal(ssm_custom._mean, np.ma.mean(vector_errors, axis=0))
        np.testing.assert_array_equal(ssm_custom._cov, np.ma.cov(vector_errors, rowvar=False, ddof=1))
        np.testing.assert_array_equal(ssm_custom._inv_cov, np.linalg.inv(np.ma.cov(vector_errors, rowvar=False, ddof=1)))

        # Check learning threshold with gaussian method
        ssm_gaussian._compute_mean_and_cov(point_errors)
        ssm_mahalanobis._compute_mean_and_cov(point_errors)
        ssm_gaussian._learn_threshold(point_errors)
        ssm_mahalanobis._learn_threshold(point_errors)
        self.assertEqual(ssm_gaussian._threshold, np.max(ssm_gaussian._compute_scores(point_errors)))
        self.assertEqual(ssm_mahalanobis._threshold, np.max(ssm_mahalanobis._compute_scores(point_errors)))

        # Check learning threshold with mahalanobis method
        ssm_gaussian._compute_mean_and_cov(vector_errors)
        ssm_mahalanobis._compute_mean_and_cov(vector_errors)
        ssm_gaussian._learn_threshold(vector_errors)
        ssm_mahalanobis._learn_threshold(vector_errors)
        self.assertEqual(ssm_gaussian._threshold, np.max(ssm_gaussian._compute_scores(vector_errors)))
        self.assertEqual(ssm_mahalanobis._threshold, np.max(ssm_mahalanobis._compute_scores(vector_errors)))

    def test_save_and_load(self):
        ssm = TSAErrorBasedChild()
        point_errors = ssm._compute_errors(self.gt_points, self.pr_points)
        vector_errors = ssm._compute_errors(self.gt_vectors, self.pr_vectors)

        # test callable are not saved using pickle
        with TemporaryDirectory() as temp_dir:
            callable_ssm = TSAErrorBasedChild(error_method="custom", error_function=lambda x: x)
            callable_ssm.save(temp_dir)

            new_ssm = TSAErrorBasedChild()
            new_ssm.load(temp_dir)

            self.assertIsNone(new_ssm.error_function)

        # check save and load with scalars
        with TemporaryDirectory() as temp_dir:
            ssm._compute_mean_and_cov(point_errors)
            ssm._learn_threshold(point_errors)

            ssm.save(temp_dir)
            contents = os.listdir(temp_dir)
            self.assertNotEqual(len(contents), 0)
            self.assertEqual(len(contents), 2)
            self.assertIn(ssm._TSAErrorBased__json_file, contents)
            self.assertIn(ssm._TSAErrorBased__numpy_file, contents)

            new_ssm = TSAErrorBasedChild()
            new_ssm.load(temp_dir)

            self.assertEqual(new_ssm.error_method, ssm.error_method)
            self.assertEqual(new_ssm.error_function, ssm.error_function)
            self.assertEqual(new_ssm.threshold_computation, ssm.threshold_computation)
            self.assertEqual(new_ssm.threshold_function, ssm.threshold_function)
            self.assertEqual(new_ssm.scoring_function, ssm.scoring_function)

            np.testing.assert_array_equal(new_ssm._mean, ssm._mean)
            np.testing.assert_array_equal(new_ssm._cov, ssm._cov)
            np.testing.assert_array_equal(new_ssm._inv_cov, ssm._inv_cov)
            self.assertEqual(new_ssm._threshold, ssm._threshold)

        # check save and load with scalars
        with TemporaryDirectory() as temp_dir:
            ssm._compute_mean_and_cov(vector_errors)
            ssm._learn_threshold(vector_errors)

            ssm.save(temp_dir)
            contents = os.listdir(temp_dir)
            self.assertNotEqual(len(contents), 0)
            self.assertEqual(len(contents), 2)
            self.assertIn(ssm._TSAErrorBased__json_file, contents)
            self.assertIn(ssm._TSAErrorBased__numpy_file, contents)

            new_ssm = TSAErrorBasedChild()
            new_ssm.load(temp_dir)

            self.assertEqual(new_ssm.error_method, ssm.error_method)
            self.assertEqual(new_ssm.error_function, ssm.error_function)
            self.assertEqual(new_ssm.threshold_computation, ssm.threshold_computation)
            self.assertEqual(new_ssm.threshold_function, ssm.threshold_function)
            self.assertEqual(new_ssm.scoring_function, ssm.scoring_function)

            np.testing.assert_array_equal(new_ssm._mean, ssm._mean)
            np.testing.assert_array_equal(new_ssm._cov, ssm._cov)
            np.testing.assert_array_equal(new_ssm._inv_cov, ssm._inv_cov)
            self.assertEqual(new_ssm._threshold, ssm._threshold)
