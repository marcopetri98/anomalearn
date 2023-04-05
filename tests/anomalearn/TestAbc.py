import unittest

from anomalearn import EqualityABC, ObtainableABC
from anomalearn.abc import FullyRepresentableABC, RepresentableABC, StringableABC
from tests.anomalearn.stubs.AbstractObjects import ObjectWithEquality, ObjectWithStr, ObjectWithStrDuckTyped, ObjectWithStrRepr, ObjectWithStrReprDuckTyped, \
    ObjectWithoutEquality, ObjectNoMoreWithEquality, ObjectWithoutEquality2, \
    ObjectWithEqualityInherit, ObjectWithNothing, ObtainableObject, \
    NotCompleteObtainable, FinallyObtainable, ObtainableChild, ObjectWithRepr, \
    ObjectWithReprDuckTyped


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
        self.assertFalse(issubclass(NotCompleteObtainable, ObtainableABC))
        self.assertTrue(issubclass(FinallyObtainable, ObtainableABC))
        
        dim = 10
        obtainable = ObtainableChild(10)
        _ = iter(obtainable)
        _ = reversed(obtainable)
        
        for idx, val in enumerate(obtainable):
            self.assertEqual(obtainable[idx], val)
            
        for idx, val in enumerate(reversed(obtainable)):
            self.assertEqual(obtainable[dim - 1 - idx], val)

    def test_stringable_abc(self):
        self.assertTrue(issubclass(ObjectWithStr, StringableABC))
        self.assertTrue(issubclass(ObjectWithStrDuckTyped, StringableABC))
        self.assertFalse(issubclass(ObjectWithNothing, StringableABC))

    def test_representable_abc(self):
        self.assertTrue(issubclass(ObjectWithRepr, RepresentableABC))
        self.assertTrue(issubclass(ObjectWithReprDuckTyped, RepresentableABC))
        self.assertFalse(issubclass(ObjectWithNothing, RepresentableABC))

    def test_fully_representable_abc(self):
        self.assertFalse(issubclass(ObjectWithStr, FullyRepresentableABC))
        self.assertFalse(issubclass(ObjectWithStrDuckTyped, FullyRepresentableABC))
        self.assertFalse(issubclass(ObjectWithRepr, FullyRepresentableABC))
        self.assertFalse(issubclass(ObjectWithReprDuckTyped, FullyRepresentableABC))
        self.assertFalse(issubclass(ObjectWithNothing, FullyRepresentableABC))
        self.assertTrue(issubclass(ObjectWithStrRepr, FullyRepresentableABC))
        self.assertTrue(issubclass(ObjectWithStrReprDuckTyped, FullyRepresentableABC))
