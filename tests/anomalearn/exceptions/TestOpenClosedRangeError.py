import unittest

import numpy as np

from anomalearn.exceptions import OpenClosedRangeError, RangeError


class TestOpenClosedRangeError(unittest.TestCase):
    def test_raise(self):
        try:
            raise OpenClosedRangeError(0, 50, np.inf)
        except OpenClosedRangeError as e:
            self.assertIsInstance(e, OpenClosedRangeError)
            self.assertFalse(e.left)
            self.assertTrue(e.right)
        
        try:
            raise OpenClosedRangeError(0, 50, np.inf)
        except RangeError as e:
            self.assertIsInstance(e, RangeError)
        
        try:
            raise OpenClosedRangeError(0, 50, np.inf)
        except Exception as e:
            self.assertIsInstance(e, Exception)
