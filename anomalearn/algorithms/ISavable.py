import abc
from abc import ABC
from typing import Any


class ISavable(ABC):
    """Interface for all objects whose parameters can be saved to file.
    """
    
    @abc.abstractmethod
    def save(self, path,
             *args,
             **kwargs) -> Any:
        """Saves the objects state.
        
        Parameters
        ----------
        path : path-like
            It is the path of the folder in which the object will be saved.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        self
            Instance to itself to allow chain calls.
    
        Raises
        ------
        ValueError
            If the given path points to an existing file and not to a directory.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self, path: str,
             *args,
             **kwargs) -> Any:
        """Loads all the parameters of the model.
        
        Parameters
        ----------
        path : str
            It is the path of the directory in which the object has been saved.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        self
            Instance to itself to allow chain calls.
    
        Raises
        ------
        ValueError
            If the given path does not point to a saved model.
        """
        raise NotImplementedError
