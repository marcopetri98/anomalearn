import unittest

import numpy as np

from anomalearn.exceptions import ClosedRangeError, RangeError


class TestClosedRangeError(unittest.TestCase):
    def test_raise(self):
        try:
            raise ClosedRangeError(0, 50, np.inf)
        except ClosedRangeError as e:
            self.assertIsInstance(e, ClosedRangeError)
            self.assertTrue(e.left)
            self.assertTrue(e.right)
        
        try:
            raise ClosedRangeError(0, 50, np.inf)
        except RangeError as e:
            self.assertIsInstance(e, RangeError)
        
        try:
            raise ClosedRangeError(0, 50, np.inf)
        except Exception as e:
            self.assertIsInstance(e, Exception)
