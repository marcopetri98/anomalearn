import abc

from anomalearn import ObtainableABC


class ObjectWithNothing(abc.ABC):
    def __init__(self):
        super().__init__()


class ObjectWithEquality(ObjectWithNothing):
    def __init__(self):
        super().__init__()
    
    def __eq__(self, other):
        raise NotImplementedError
    
    def __ne__(self, other):
        raise NotImplementedError


class ObjectNoMoreWithEquality(ObjectWithEquality):
    def __init__(self):
        super().__init__()
    
    __eq__ = None


class ObjectWithoutEquality(ObjectWithNothing):
    def __init__(self):
        super().__init__()
    
    def __eq__(self, other):
        raise NotImplementedError


class ObjectWithoutEquality2(ObjectWithoutEquality):
    def __init__(self):
        super().__init__()


class ObjectWithEqualityInherit(ObjectWithoutEquality2):
    def __init__(self):
        super().__init__()
    
    def __ne__(self, other):
        raise NotImplementedError


class ObtainableObject(ObjectWithNothing):
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, item):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError
    
    def __reversed__(self):
        raise NotImplementedError


class NoMoreObtainableObject(ObjectWithNothing):
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, item):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError
    
    __reversed__ = None
    
    
class NotCompleteObtainable(ObjectWithNothing):
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, item):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    

class FinallyObtainable(NotCompleteObtainable):
    def __init__(self):
        super().__init__()
    
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
