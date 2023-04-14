import unittest

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import ExtraTreeRegressor
import numpy as np

from anomalearn.utils import are_normal_attr_equal, are_numpy_attr_equal
from anomalearn.utils.scikit import (are_list_attr_equal,
                                     are_random_state_attr_equal,
                                     are_tree_attr_equal)


class TestScikitFunction(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.series = rng.random((100, 3))
        self.fake_out = rng.random((100, 1))
    
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

    def test_are_list_attr_equal(self):
        isof1 = IsolationForest(random_state=11)
        isof2 = IsolationForest(random_state=11)
        list_attr = ["estimators_features_", "estimators_samples_"]

        isof1.fit(self.series)
        isof2.fit(self.series)
        self.assertTrue(are_list_attr_equal(isof1, isof1, list_attr))
        self.assertTrue(are_list_attr_equal(isof1, isof2, list_attr))
        self.assertTrue(are_list_attr_equal(isof2, isof1, list_attr))

        isof2 = IsolationForest(random_state=121)
        isof2.fit(self.fake_out)
        self.assertFalse(are_list_attr_equal(isof1, isof2, list_attr))
        self.assertFalse(are_list_attr_equal(isof2, isof1, list_attr))

    def test_are_random_state_attr_equal(self):
        xtr1 = ExtraTreeRegressor()
        xtr2 = ExtraTreeRegressor()
        rand_attr = ["random_state"]

        self.assertTrue(are_random_state_attr_equal(xtr1, xtr1, rand_attr))
        self.assertTrue(are_random_state_attr_equal(xtr1, xtr2, rand_attr))
        self.assertTrue(are_random_state_attr_equal(xtr2, xtr1, rand_attr))

        xtr2 = ExtraTreeRegressor(random_state=11)
        self.assertFalse(are_random_state_attr_equal(xtr1, xtr2, rand_attr))
        self.assertFalse(are_random_state_attr_equal(xtr2, xtr1, rand_attr))

        xtr1 = ExtraTreeRegressor(random_state=11)
        self.assertTrue(are_random_state_attr_equal(xtr1, xtr1, rand_attr))
        self.assertTrue(are_random_state_attr_equal(xtr1, xtr2, rand_attr))
        self.assertTrue(are_random_state_attr_equal(xtr2, xtr1, rand_attr))

        xtr2 = ExtraTreeRegressor(random_state=np.random.RandomState(22))
        self.assertFalse(are_random_state_attr_equal(xtr1, xtr2, rand_attr))
        self.assertFalse(are_random_state_attr_equal(xtr2, xtr1, rand_attr))

        xtr1 = ExtraTreeRegressor(random_state=np.random.RandomState(22))
        self.assertTrue(are_random_state_attr_equal(xtr1, xtr1, rand_attr))
        self.assertTrue(are_random_state_attr_equal(xtr1, xtr2, rand_attr))
        self.assertTrue(are_random_state_attr_equal(xtr2, xtr1, rand_attr))

    def test_are_tree_attr_equal(self):
        xtr1 = ExtraTreeRegressor(random_state=11)
        xtr2 = ExtraTreeRegressor(random_state=11)
        tree_attr = ["tree_"]

        xtr1.fit(self.series, self.fake_out)
        xtr2.fit(self.series, self.fake_out)
        self.assertTrue(are_tree_attr_equal(xtr1, xtr1, tree_attr))
        self.assertTrue(are_tree_attr_equal(xtr1, xtr2, tree_attr))
        self.assertTrue(are_tree_attr_equal(xtr2, xtr1, tree_attr))

        xtr2 = ExtraTreeRegressor(random_state=22)
        xtr2.fit(self.series, self.fake_out)
        self.assertFalse(are_tree_attr_equal(xtr1, xtr2, tree_attr))
        self.assertFalse(are_tree_attr_equal(xtr2, xtr1, tree_attr))
