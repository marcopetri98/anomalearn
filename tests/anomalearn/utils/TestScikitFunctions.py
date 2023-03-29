import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from anomalearn.utils import are_numpy_attr_equal, are_normal_attr_equal


class TestScikitFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.series = np.random.rand(100, 3)
    
    def test_are_numpy_attr_equal(self):
        minmax1 = MinMaxScaler()
        minmax2 = MinMaxScaler()
        minmax_numpy = ["min_", "scale_", "data_min_", "data_max_",
                        "data_range_", "n_features_in_", "n_samples_seen_"]
        
        self.assertTrue(are_numpy_attr_equal(minmax1, minmax2, minmax_numpy))
        minmax1.fit(self.series)
        self.assertFalse(are_numpy_attr_equal(minmax1, minmax2, minmax_numpy))
        self.assertFalse(are_numpy_attr_equal(minmax2, minmax1, minmax_numpy))
        minmax2.fit(self.series)
        self.assertTrue(are_numpy_attr_equal(minmax1, minmax2, minmax_numpy))
    
    def test_are_normal_attr_equal(self):
        minmax1 = MinMaxScaler()
        minmax2 = MinMaxScaler()
        normal_attr = ["feature_range", "copy", "clip"]
        
        self.assertTrue(are_normal_attr_equal(minmax1, minmax2, normal_attr))
        minmax1 = MinMaxScaler(copy=False)
        self.assertFalse(are_normal_attr_equal(minmax1, minmax2, normal_attr))
        self.assertFalse(are_normal_attr_equal(minmax2, minmax1, normal_attr))
        minmax2 = MinMaxScaler(copy=False)
        self.assertTrue(are_normal_attr_equal(minmax1, minmax2, normal_attr))
