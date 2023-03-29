import unittest

import numpy as np

from anomalearn.utils import get_rows_without_nan


class TestArraysFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = np.random.rand(100)
    
    def test_get_rows_without_nan(self):
        not_nan_rows = get_rows_without_nan(self.points)
        np.testing.assert_array_equal(np.arange(self.points.shape[0]), not_nan_rows)

        new_points = self.points.copy()
        new_points[0] = np.nan
        not_nan_rows = get_rows_without_nan(new_points)
        self.assertNotIn(0, not_nan_rows)

        new_points = self.points.copy()
        new_points[0] = np.nan
        new_points[2] = np.nan
        new_points = new_points.reshape(-1, 2)
        not_nan_rows = get_rows_without_nan(new_points)
        self.assertNotIn(0, not_nan_rows)
        self.assertNotIn(1, not_nan_rows)
