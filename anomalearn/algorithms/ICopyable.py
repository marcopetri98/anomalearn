from __future__ import annotations

import abc


class ICopyable(abc.ABC):
    """Interface for objects that can be copied.
    """
    @abc.abstractmethod
    def copy(self) -> ICopyable:
        """Copies the object.
        
        Returns
        -------
        obj_copy : ICopyable
            It is the copy of the object.
        """
        raise NotImplementedError
