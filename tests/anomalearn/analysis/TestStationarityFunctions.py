import unittest

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from anomalearn.analysis import analyse_stationarity


class TestStationarityFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng()
        cls.series = rng.random(1000, dtype=np.double)
        
    def assert_stationarity_test(self, expected, actual):
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertDictEqual(expected[2], actual[2])
        
    def test_analyse_stationarity(self):
        statistic, pvalue, critics = analyse_stationarity(self.series, "adfuller")
        statistic_, pvalue_, _, _, critics_, _ = adfuller(self.series)
        self.assert_stationarity_test([statistic_, pvalue_, critics_], [statistic, pvalue, critics])
        
        statistic, pvalue, critics = analyse_stationarity(self.series, "adfuller", diff_order=1)
        statistic_, pvalue_, _, _, critics_, _ = adfuller(np.diff(self.series, 1))
        self.assert_stationarity_test([statistic_, pvalue_, critics_], [statistic, pvalue, critics])
        
        statistic, pvalue, critics = analyse_stationarity(self.series, "adfuller", {"maxlag": 4})
        statistic_, pvalue_, _, _, critics_, _ = adfuller(self.series, maxlag=4)
        self.assert_stationarity_test([statistic_, pvalue_, critics_], [statistic, pvalue, critics])
        
        statistic, pvalue, critics = analyse_stationarity(self.series, "kpss", diff_order=1)
        statistic_, pvalue_, _, critics_ = kpss(np.diff(self.series, 1))
        self.assert_stationarity_test([statistic_, pvalue_, critics_], [statistic, pvalue, critics])
        
        statistic, pvalue, critics = analyse_stationarity(self.series, "kpss")
        statistic_, pvalue_, _, critics_ = kpss(self.series)
        self.assert_stationarity_test([statistic_, pvalue_, critics_], [statistic, pvalue, critics])
        
        statistic, pvalue, critics = analyse_stationarity(self.series, "kpss", {"nlags": 4})
        statistic_, pvalue_, _, critics_ = kpss(self.series, nlags=4)
        self.assert_stationarity_test([statistic_, pvalue_, critics_], [statistic, pvalue, critics])
        