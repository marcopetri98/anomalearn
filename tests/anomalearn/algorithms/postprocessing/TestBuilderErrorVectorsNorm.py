import unittest

import numpy as np

from anomalearn.algorithms.postprocessing import BuilderErrorVectorsNorm


class TestBuilderErrorVectorsNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pred = np.random.rand(1000, 10)
        cls.true = np.random.rand(1000, 10)
    
    def test_copy(self):
        builder = BuilderErrorVectorsNorm()
        new = builder.copy()
        
        self.assertIsNot(new, builder)
        self.assertIsInstance(new, BuilderErrorVectorsNorm)
    
    def test_shape_change(self):
        builder = BuilderErrorVectorsNorm()
        errors, fake_out = builder.shape_change(self.pred, self.true)

        self.assertEqual(self.pred.shape[0], errors.shape[0])
        self.assertEqual(1, errors.shape[1])
        np.testing.assert_array_equal(np.linalg.norm(self.true - self.pred, axis=1).reshape((-1, 1)), errors)
        np.testing.assert_array_equal(np.array([]), fake_out)
