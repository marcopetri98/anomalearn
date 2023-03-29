import abc
from abc import ABC

import numpy as np


class IClassifier(ABC):
    """Interface identifying a machine learning classifier.
    """

    @abc.abstractmethod
    def classify(self, x, *args, **kwargs) -> np.ndarray:
        """Computes the labels for the given points.
        
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
        labels : ndarray
            The labels resulted from the classification. The array must have at
            least 2 dimensions in which the first is equal to the first
            dimension of `x` (usually, it has 2 dimensions and the second is 1).
        """
        raise NotImplementedError
