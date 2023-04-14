from collections.abc import Reversible, Sized
import abc
import logging


__module_logger = logging.getLogger(__name__)


def _check_methods(klass, *methods):
    """Check if the class or any of its parents has all specified methods.
    
    Between the parents that are searched, object is not considered. We consider
    that the class has a method only if at least one class more specific than
    object has the method. That is: it does not use the default implementation.
    
    Parameters
    ----------
    klass : class
        The class to be checked.
    
    methods : list[str]
        The methods that the class must have.

    Returns
    -------
    are_there_methods
        NotImplemented if there is no method or a method is None. True if all
        methods are present.
    """
    __module_logger.debug(f"class should implement methods {methods}")
    __module_logger.debug(f"received class {klass}")
    __module_logger.debug(f"class.__dict__ = {klass.__dict__}")
    # exclude object an maintain order
    mro = [e for e in klass.__mro__ if e is not object]
    for method in methods:
        for base in mro:
            if method in base.__dict__:
                if base.__dict__[method] is None:
                    return NotImplemented
                else:
                    __module_logger.debug(f"method {method} implemented")
                    break
        else:
            return NotImplemented
    return True


class EqualityABC(abc.ABC):
    """Abstract class for objects implementing == and !=.
    """
    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError
    
    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is EqualityABC:
            return _check_methods(C, "__eq__", "__ne__")
        return NotImplemented


class RepresentableABC(abc.ABC):
    """Abstract class for objects implementing __repr__."""
    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is RepresentableABC:
            return _check_methods(C, "__repr__")
        return NotImplemented
    
    
class StringableABC(abc.ABC):
    """Abstract class for objects implementing __str__."""
    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is StringableABC:
            return _check_methods(C, "__str__")
        return NotImplemented
    
    
class FullyRepresentableABC(RepresentableABC, StringableABC):
    """Abstract class for objects implementing __str__ and __repr__."""
    @classmethod
    def __subclasshook__(cls, C):
        if cls is FullyRepresentableABC:
            return _check_methods(C, "__repr__", "__str__")
        return NotImplemented


class ObtainableABC(Reversible, Sized):
    """Abstract class for objects implementing __getitem__.
    """
    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    def __iter__(self):
        i = 0
        while i < len(self):
            value = self[i]
            yield value
            i += 1
    
    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]

    @classmethod
    def __subclasshook__(cls, C):
        if cls is ObtainableABC:
            return _check_methods(C, "__getitem__", "__len__", "__iter__")
        return NotImplemented
