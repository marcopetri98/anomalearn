import unittest

from anomalearn.exceptions import SelectionError


class TestSelectionError(unittest.TestCase):
    def test_raise(self):
        try:
            raise SelectionError(["a", "b", "c"], "Desmond Miles")
        except SelectionError as e:
            self.assertIsInstance(e, SelectionError)
        
        try:
            raise SelectionError(["a", "b", "c"], "Desmond Miles")
        except Exception as e:
            self.assertIsInstance(e, Exception)
