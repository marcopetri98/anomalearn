import unittest

import numpy as np

from anomalearn.exceptions import ClosedOpenRangeError, RangeError


class TestClosedOpenRangeError(unittest.TestCase):
    def test_raise(self):
        try:
            raise ClosedOpenRangeError(0, 50, np.inf)
        except ClosedOpenRangeError as e:
            self.assertIsInstance(e, ClosedOpenRangeError)
            self.assertTrue(e.left)
            self.assertFalse(e.right)
        
        try:
            raise ClosedOpenRangeError(0, 50, np.inf)
        except RangeError as e:
            self.assertIsInstance(e, RangeError)
        
        try:
            raise ClosedOpenRangeError(0, 50, np.inf)
        except Exception as e:
            self.assertIsInstance(e, Exception)
