import unittest

import numpy as np

from anomalearn.exceptions import OpenRangeError, RangeError


class TestOpenRangeError(unittest.TestCase):
    def test_raise(self):
        try:
            raise OpenRangeError(-np.inf, np.inf, np.inf)
        except OpenRangeError as e:
            self.assertIsInstance(e, OpenRangeError)
            self.assertFalse(e.left)
            self.assertFalse(e.right)
            
        try:
            raise OpenRangeError(-np.inf, np.inf, np.inf)
        except RangeError as e:
            self.assertIsInstance(e, RangeError)
        
        try:
            raise OpenRangeError(-np.inf, np.inf, np.inf)
        except Exception as e:
            self.assertIsInstance(e, Exception)
