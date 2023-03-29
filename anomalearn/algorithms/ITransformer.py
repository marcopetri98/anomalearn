import abc

import numpy as np


class ITransformer(abc.ABC):
    """The interface for objects implementing transformations.
    """
    
    @abc.abstractmethod
    def transform(self, x, *args, **kwargs) -> np.ndarray:
        """Transforms the input.
        
        Parameters
        ----------
        x : array-like
            The data to transform. Data must have at least two dimensions in
            which the first dimension represent the number of samples.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        transformations : ndarray with shape[0]=x.shape[0]
            The transformations for the points in input.
        """
        raise NotImplementedError
