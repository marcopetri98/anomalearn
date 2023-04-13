import unittest

from anomalearn.exceptions import SelectionError


class TestSelectionError(unittest.TestCase):
    def test_raise(self):
        try:
            raise SelectionError(["a", "b", "c"], "Desmond Miles")
        except SelectionError as e:
            self.assertIsInstance(e, SelectionError)
