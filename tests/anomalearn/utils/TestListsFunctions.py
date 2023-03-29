import unittest

import numpy as np

from anomalearn.utils import all_indices, concat_list_array


class TestListsFunctions(unittest.TestCase):
    def test_all_indices(self):
        list_ = list(range(10))
        
        self.assertEqual([0], all_indices(list_, 0))
        self.assertEqual([5], all_indices(list_, 5))
        self.assertEqual([9], all_indices(list_, 9))
        self.assertEqual([], all_indices(list_, 10))
        
        list_[0], list_[4], list_[6], list_[9] = 100, 100, 100, 100
        self.assertEqual([0, 4, 6, 9], all_indices(list_, 100))
    
    def test_concat_list_arrays(self):
        arr1 = np.random.rand(100, 3)
        arr2 = np.random.rand(100, 3)
        arr3 = np.random.rand(100, 3)
        
        np.testing.assert_array_equal(np.concatenate((arr1, arr2, arr3)), concat_list_array([arr1, arr2, arr3]))
        np.testing.assert_array_equal(np.concatenate((arr1, arr2)), concat_list_array([arr1, arr2]))
        np.testing.assert_array_equal(arr1, concat_list_array([arr1]))
        np.testing.assert_array_equal(np.array([]), concat_list_array([]))
