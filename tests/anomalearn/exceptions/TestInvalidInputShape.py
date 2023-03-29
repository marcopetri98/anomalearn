import unittest

from anomalearn.exceptions import InvalidInputShape


class TestInvalidInputShape(unittest.TestCase):
    def test_raise(self):
        try:
            raise InvalidInputShape((10, 3), tuple([10]))
        except InvalidInputShape as e:
            self.assertIsInstance(e, InvalidInputShape)
            
        try:
            raise InvalidInputShape((10, 3), tuple([10]))
        except Exception as e:
            self.assertIsInstance(e, Exception)
        