import abc

import numpy as np


class IPredictor(abc.ABC):
    """Interface identifying a machine learning predictor.
    """
    
    @abc.abstractmethod
    def predict(self, x, *args, **kwargs) -> np.ndarray:
        """Computes prediction for each data point in input.
        
        Parameters
        ----------
        x : array-like
            The data used for prediction. Data must have at least two dimensions
            in which the first dimension represent the number of samples.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        predictions : ndarray
            The predictions for the points in input. The predictions must have
            at least two dimensions in which the first is the number of samples,
            and it is identical to the first dimension of `x`.
        """
        raise NotImplementedError
