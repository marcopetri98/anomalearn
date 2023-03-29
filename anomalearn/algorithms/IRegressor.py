import abc
from abc import ABC

import numpy as np


class IRegressor(ABC):
    """Interface identifying a machine learning regressor.
    """
    
    @abc.abstractmethod
    def regress(self, x, *args, **kwargs) -> np.ndarray:
        """Computes regression given the points X.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The array of the samples for which we need to regress the quantity
            estimated by the model.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        regression_value : array-like of shape (n_samples, n_out_features)
            The regression value given the input from which the model must
            perform regression.
        """
        raise NotImplementedError
