import unittest

from anomalearn import EqualityABC, ObtainableABC
from tests.anomalearn.stubs.AbstractObjects import ObjectWithEquality, \
    ObjectWithoutEquality, ObjectNoMoreWithEquality, ObjectWithoutEquality2, \
    ObjectWithEqualityInherit, ObjectWithNothing, ObtainableObject, \
    NoMoreObtainableObject, NotCompleteObtainable, FinallyObtainable, \
    ObtainableChild


class TestAbc(unittest.TestCase):
    def test_equality_abc(self):
        self.assertFalse(issubclass(ObjectWithNothing, EqualityABC))
        self.assertTrue(issubclass(ObjectWithEquality, EqualityABC))
        self.assertFalse(issubclass(ObjectNoMoreWithEquality, EqualityABC))
        self.assertFalse(issubclass(ObjectWithoutEquality, EqualityABC))
        self.assertFalse(issubclass(ObjectWithoutEquality2, EqualityABC))
        self.assertTrue(issubclass(ObjectWithEqualityInherit, EqualityABC))

    def test_obtainable_abc(self):
        self.assertFalse(issubclass(ObjectWithNothing, ObtainableABC))
        self.assertTrue(issubclass(ObtainableObject, ObtainableABC))
        self.assertFalse(issubclass(NoMoreObtainableObject, ObtainableABC))
        self.assertFalse(issubclass(NotCompleteObtainable, ObtainableABC))
        self.assertTrue(issubclass(FinallyObtainable, ObtainableABC))
        
        dim = 10
        obtainable = ObtainableChild(10)
        iterator = iter(obtainable)
        reverse_iterator = reversed(obtainable)
        
        for idx, val in enumerate(obtainable):
            self.assertEqual(obtainable[idx], val)
            
        for idx, val in enumerate(reversed(obtainable)):
            self.assertEqual(obtainable[dim - 1 - idx], val)
