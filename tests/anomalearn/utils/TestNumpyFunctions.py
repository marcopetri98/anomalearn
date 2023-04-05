import unittest

from numpy.random import RandomState

from anomalearn.utils.numpy import are_random_state_equal


class TestNumpyFunctions(unittest.TestCase):
    def test_are_random_state_equal(self):
        rs1 = RandomState(101)
        rs2 = RandomState(101)

        self.assertTrue(are_random_state_equal(rs1, rs2))

        _ = rs1.random_sample(1)

        self.assertFalse(are_random_state_equal(rs1, rs2))
