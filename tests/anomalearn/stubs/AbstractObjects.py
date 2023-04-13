import abc

from anomalearn import ObtainableABC
from anomalearn.abc import FullyRepresentableABC, RepresentableABC, StringableABC


class ObjectWithNothing(abc.ABC):
    pass


#################################################
#                                               #
#                                               #
#        STUBS FOR THE EQUALITY CLASS           #
#                                               #
#                                               #
#################################################
class ObjectWithEquality(ObjectWithNothing):
    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError


class ObjectNoMoreWithEquality(ObjectWithEquality):
    __eq__ = None


class ObjectWithoutEquality(ObjectWithNothing):
    def __eq__(self, other):
        raise NotImplementedError


class ObjectWithoutEquality2(ObjectWithoutEquality):
    pass


class ObjectWithEqualityInherit(ObjectWithoutEquality2):
    def __ne__(self, other):
        raise NotImplementedError


#################################################
#                                               #
#                                               #
#        STUBS FOR THE STRINGABLE CLASS         #
#                                               #
#                                               #
#################################################
class ObjectWithStr(StringableABC):
    pass


class ObjectWithStrDuckTyped:
    def __str__(self) -> str:
        return ""


#################################################
#                                               #
#                                               #
#        STUBS FOR THE REPRESENTABLE CLASS      #
#                                               #
#                                               #
#################################################
class ObjectWithRepr(RepresentableABC):
    pass


class ObjectWithReprDuckTyped:
    def __repr__(self) -> str:
        return ""


#################################################
#                                               #
#                                               #
#    STUBS FOR THE FULLY REPRESENTABLE CLASS    #
#                                               #
#                                               #
#################################################
class ObjectWithStrRepr(FullyRepresentableABC):
    pass


class ObjectWithStrReprDuckTyped:
    def __str__(self) -> str:
        return ""
    
    def __repr__(self) -> str:
        return ""


#################################################
#                                               #
#                                               #
#        STUBS FOR THE OBTAINABLE CLASS         #
#                                               #
#                                               #
#################################################

class ObtainableObject(ObjectWithNothing):
    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __reversed__(self):
        raise NotImplementedError


class NotCompleteObtainable(ObjectWithNothing):
    def __getitem__(self, item):
        raise NotImplementedError


class FinallyObtainable(NotCompleteObtainable):
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __reversed__(self):
        raise NotImplementedError


class ObtainableChild(ObtainableABC):
    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim

    def __getitem__(self, item):
        if not 0 <= item < self.dim:
            raise IndexError
        
        return item

    def __len__(self):
        return self.dim
