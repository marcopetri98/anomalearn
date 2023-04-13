import abc

import numpy as np

from . import IClassifier


class IBoundaryClassifier(IClassifier):
    """Interface identifying a machine learning classifier computing a boundary.
    """
    
    @abc.abstractmethod
    def decision_function(self, x, *args, **kwargs) -> np.ndarray:
        """Compute the value of the decision function for each point of `x`.
        
        Parameters
        ----------
        x : array-like
            The data to be classified. Data must have at least two dimensions in
            which the first dimension represent the number of samples.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        values : ndarray
            The values of the decision function for each point in input.
        """
        raise NotImplementedError
