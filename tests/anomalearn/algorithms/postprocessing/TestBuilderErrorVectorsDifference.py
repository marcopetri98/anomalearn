import unittest

import numpy as np

from anomalearn.algorithms.postprocessing import BuilderErrorVectorsDifference


class TestBuilderErrorVectorsDifference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pred = np.random.rand(1000, 10)
        cls.true = np.random.rand(1000, 10)
    
    def test_copy(self):
        builder = BuilderErrorVectorsDifference()
        new = builder.copy()
        
        self.assertIsNot(new, builder)
        self.assertIsInstance(new, BuilderErrorVectorsDifference)
    
    def test_shape_change(self):
        builder = BuilderErrorVectorsDifference()
        errors, fake_out = builder.shape_change(self.pred, self.true)
        
        self.assertTupleEqual(self.pred.shape, errors.shape)
        np.testing.assert_array_equal(self.true - self.pred, errors)
        np.testing.assert_array_equal(np.array([]), fake_out)
        