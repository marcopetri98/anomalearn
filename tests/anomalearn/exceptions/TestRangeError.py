import unittest

import numpy as np

from anomalearn.exceptions import RangeError


class TestRangeError(unittest.TestCase):
    def test_raise(self):
        try:
            raise RangeError(False, -np.inf, np.inf, False, np.inf)
        except RangeError as e:
            self.assertIsInstance(e, RangeError)
        
        try:
            raise RangeError(False, -np.inf, np.inf, False, np.inf)
        except Exception as e:
            self.assertIsInstance(e, Exception)
