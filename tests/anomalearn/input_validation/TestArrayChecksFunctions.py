import unittest

import numpy as np
from sklearn.utils import check_array

from anomalearn.input_validation import check_array_general, check_array_1d, \
    check_x_y_smaller_1d


class TestArrayChecksFunctions(unittest.TestCase):
    def test_check_array_general(self):
        array1 = np.random.rand(50, 3)
        array2 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        array3 = np.random.rand(10, 10, 10, 10)
        
        # check that nothing is raised
        check_array_general(array1, 2, None, True)
        check_array_general(array2, 2, None, True)
        check_array_general(array3, 4, None, True)
        
        # if something is not array-like throw ValueError
        not_array = "Does not expose array interface"
        self.assertRaises(ValueError, check_array_general, not_array, 2, None, True)
        self.assertRaises(ValueError, check_array_general, object(), 2, None, True)
        self.assertRaises(ValueError, check_array_general, "Frodo Baggins", 2, None, True)
        
        # if number of dimensions is wrong throw ValueError
        self.assertRaises(ValueError, check_array_general, array1, 7, None, True)
        
        # if minimum number of samples is wrong throw ValueError
        self.assertRaises(ValueError, check_array_general, array1, 2, (100, 10), True)
        
        # check force_all_finite attribute
        array1[5, 0] = np.nan
        self.assertRaises(ValueError, check_array_general, array1, 2, None, True)
        check_array_general(array1, 2, None, "allow-nan")
        array1[5, 0] = np.inf
        self.assertRaises(ValueError, check_array_general, array1, 2, None, "allow-nan")
        check_array_general(array1, 2, None, False)
    
    def test_check_array_1d(self):
        array1 = np.random.rand(20)
        array2 = [1, 1, 1, 1, 1]
        
        # check that nothing is raised
        check_array_1d(array1)
        check_array_1d(array2)
        
        # if something is not array-like or not 1d throw ValueError
        not_array = "Does not expose array interface"
        not_array1d = [[1, 1], [1, 1], [1, 1], [1, 1]]
        self.assertRaises(ValueError, check_array_1d, not_array)
        self.assertRaises(ValueError, check_array_1d, object())
        self.assertRaises(ValueError, check_array_1d, "Bilbo Baggins")
        self.assertRaises(ValueError, check_array_1d, not_array1d)
        
        # check force_all_finite attribute
        array1[10] = np.nan
        self.assertRaises(ValueError, check_array_1d, array1)
        check_array_1d(array1, "allow-nan")
        array1[10] = np.inf
        self.assertRaises(ValueError, check_array_1d, array1, "allow-nan")
        check_array_1d(array1, False)
    
    def test_check_x_y_smaller_1d(self):
        array1 = np.random.rand(20)
        array2 = [1, 1, 1, 1, 1]
        array3 = np.random.rand(30)
        
        # check that nothing is raised
        check_x_y_smaller_1d(array1, array1)
        check_x_y_smaller_1d(array2, array2)
        check_x_y_smaller_1d(array1, array3)

        # if something is not array-like or not 1d throw ValueError
        not_array = "Does not expose array interface"
        not_array1d = [[1, 1], [1, 1], [1, 1], [1, 1]]
        self.assertRaises(ValueError, check_x_y_smaller_1d, array1, not_array)
        self.assertRaises(ValueError, check_x_y_smaller_1d, not_array, array1)
        self.assertRaises(ValueError, check_x_y_smaller_1d, array1, object())
        self.assertRaises(ValueError, check_x_y_smaller_1d, object(), array1)
        self.assertRaises(ValueError, check_x_y_smaller_1d, array1, "Bilbo Baggins")
        self.assertRaises(ValueError, check_x_y_smaller_1d, "Bilbo Baggins", array1)
        self.assertRaises(ValueError, check_x_y_smaller_1d, array1, not_array1d)
        self.assertRaises(ValueError, check_x_y_smaller_1d, not_array1d, array1)
        
        # check that x is smaller or equal to y
        self.assertRaises(ValueError, check_x_y_smaller_1d, array3, array1)

        # check force_all_finite attribute
        array1[10] = np.nan
        self.assertRaises(ValueError, check_array_1d, array1, array3)
        self.assertRaises(ValueError, check_array_1d, array1, array1)
        check_x_y_smaller_1d(array1, array3, "allow-nan")
        check_x_y_smaller_1d(array1, array1, "allow-nan")
        array1[10] = np.inf
        self.assertRaises(ValueError, check_array_1d, array1, array3, "allow-nan")
        self.assertRaises(ValueError, check_array_1d, array1, array1, "allow-nan")
        check_x_y_smaller_1d(array1, array3, False)
        check_x_y_smaller_1d(array1, array1, False)
        
        array3[10] = np.nan
        self.assertRaises(ValueError, check_array_1d, array2, array3)
        check_x_y_smaller_1d(array2, array3, "allow-nan")
        array3[10] = np.inf
        self.assertRaises(ValueError, check_array_1d, array2, array3, "allow-nan")
        check_x_y_smaller_1d(array2, array3, False)
