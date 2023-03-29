import unittest

import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from anomalearn.analysis import decompose_time_series


class TestDecompositionFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng()
        cls.series = rng.random(100, dtype=np.double)
    
    def assert_decompose_time_series(self, expected, actual):
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        np.testing.assert_array_equal(expected[2], actual[2])
    
    def test_decompose_time_series(self):
        trend, seasonal, residual = decompose_time_series(self.series, "stl", method_params={"period": 5})
        result = STL(self.series, period=5).fit()
        self.assert_decompose_time_series([result.trend, result.seasonal, result.resid], [trend, seasonal, residual])
        
        trend, seasonal, residual = decompose_time_series(self.series, "stl", diff_order=1, method_params={"period": 5})
        result = STL(np.diff(self.series, 1), period=5).fit()
        self.assert_decompose_time_series([result.trend, result.seasonal, result.resid], [trend, seasonal, residual])
        
        trend, seasonal, residual = decompose_time_series(self.series, "moving_average", method_params={"period": 5})
        result = seasonal_decompose(self.series, period=5)
        self.assert_decompose_time_series([result.trend, result.seasonal, result.resid], [trend, seasonal, residual])
        
        trend, seasonal, residual = decompose_time_series(self.series, "moving_average", diff_order=1, method_params={"period": 5})
        result = seasonal_decompose(np.diff(self.series, 1), period=5)
        self.assert_decompose_time_series([result.trend, result.seasonal, result.resid], [trend, seasonal, residual])
