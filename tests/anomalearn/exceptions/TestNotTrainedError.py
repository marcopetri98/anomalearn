import unittest

from anomalearn.exceptions import NotTrainedError


class TestNotTrainedError(unittest.TestCase):
    def test_raise(self):
        try:
            raise NotTrainedError()
        except NotTrainedError as e:
            self.assertIsInstance(e, NotTrainedError)
        
        try:
            raise NotTrainedError()
        except Exception as e:
            self.assertIsInstance(e, Exception)
