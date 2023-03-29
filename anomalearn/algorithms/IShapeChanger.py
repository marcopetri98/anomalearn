import abc
from typing import Tuple

import numpy as np


class IShapeChanger(abc.ABC):
    """Changes the shape of the input data and targets.
    """
    
    @abc.abstractmethod
    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Change the shape of the input data.
        
        Parameters
        ----------
        x : array-like
            The data to change in shape. Data must have at least two dimensions
            in which the first dimension represent the number of samples.
            
        y : array-like, default=None
            The target for the shape changing data. Data must have at least two
            dimensions in which the first dimension represent the number of
            samples. Moreover, if not `None`, the first dimension must be the
            same as that of `x`.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        x_new : ndarray
            It is the array of the transformed input. It must have at least two
            dimensions in which the first dimension represent the number of
            samples.
        
        y_new : ndarray
            It is the array of the transformed targets. It has at least two
            dimensions and the first is identical to that of `x_new`.
        """
        raise NotImplementedError
