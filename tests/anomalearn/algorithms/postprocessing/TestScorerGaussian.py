import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import multivariate_normal

from anomalearn.algorithms.postprocessing import ScorerGaussian
from anomalearn.utils import estimate_mean_covariance


class TestScorerGaussian(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = np.random.rand(100, 1)
        cls.vectors = np.random.rand(100, 3)
    
    def test_equality(self):
        scorer1 = ScorerGaussian()
        scorer2 = ScorerGaussian()
        
        self.assertIsNone(scorer1.mean)
        self.assertIsNone(scorer1.cov)
        self.assertEqual(scorer1, scorer2)
        
        scorer1._mean = np.array([1, 1, 1])
        self.assertNotEqual(scorer1, scorer2)
        scorer1._cov = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertNotEqual(scorer1, scorer2)
        
        scorer2._mean = scorer1._mean
        self.assertNotEqual(scorer1, scorer2)
        scorer2._cov = scorer1._cov
        self.assertEqual(scorer1, scorer2)
        
        self.assertNotEqual(scorer1, "John-117")
    
    def test_copy(self):
        scorer = ScorerGaussian()
        
        new = scorer.copy()
        self.assertEqual(scorer, new)
        self.assertIsNot(scorer, new)

        scorer._mean = np.array([1, 1, 1])
        scorer._cov = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        
        new = scorer.copy()
        self.assertEqual(scorer, new)
        self.assertIsNot(scorer, new)
    
    def test_fit(self):
        scorer = ScorerGaussian()
        
        mean, cov, _ = estimate_mean_covariance(self.points)
        scorer.fit(self.points)
        np.testing.assert_array_equal(mean, scorer.mean)
        np.testing.assert_array_equal(cov, scorer.cov)
        
        mean, cov, _ = estimate_mean_covariance(self.vectors)
        scorer.fit(self.vectors)
        np.testing.assert_array_equal(mean, scorer.mean)
        np.testing.assert_array_equal(cov, scorer.cov)
    
    def test_shape_change(self):
        scorer = ScorerGaussian()
        
        for input_ in [self.points, self.vectors]:
            mean, cov, _ = estimate_mean_covariance(input_)
            scorer.fit(input_)
            
            probas = multivariate_normal.pdf(input_, mean=mean, cov=cov, allow_singular=True) + 1e-10
            scores, _ = scorer.shape_change(input_)
            np.testing.assert_array_equal(1 / probas, scores)
            
            new_input = input_.copy()
            new_input[:10, :] = np.nan
            exp_scores = 1 / probas
            exp_scores[:10] = np.nan
            scores, _ = scorer.shape_change(new_input)
            np.testing.assert_array_equal(exp_scores, scores)
    
    def test_save_and_load(self):
        scorer = ScorerGaussian()
        
        with TemporaryDirectory() as tmp_dir:
            scorer.save(tmp_dir)
            
            new_scorer = ScorerGaussian()
            new_scorer._mean = np.array([1, 1, 1])
            new_scorer._cov = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            new_scorer.load(tmp_dir)
            self.assertEqual(scorer, new_scorer)

            scorer._mean = np.array([1, 1, 1])
            scorer._cov = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            scorer.save(tmp_dir)
            
            new_scorer = ScorerGaussian()
            new_scorer.load(tmp_dir)
            self.assertEqual(scorer, new_scorer)

        path = Path(__file__).parent / (str(Path(__file__).name).split(".")[0] + "_temp_.txt")
        path.touch()
        self.assertRaises(ValueError, scorer.save, str(path))
        path.unlink()
