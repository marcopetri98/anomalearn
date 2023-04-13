import abc

import numpy as np

from . import IClassifier


class IProbabilisticClassifier(IClassifier):
    """Interface identifying a machine learning probabilistic classifier.
    """
    
    @abc.abstractmethod
    def predict_proba(self, x, *args, **kwargs) -> np.ndarray:
        """Computes the probabilities for the given points.
        
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
        probabilities : ndarray
            The probabilities of points. The array must have at least 2
            dimensions in which the first is equal to the first
            dimension of `x` (usually, it has 2 dimensions and the second is
            equal to the number of features in `x`).
        """
        raise NotImplementedError
