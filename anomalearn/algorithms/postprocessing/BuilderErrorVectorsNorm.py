from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.utils import check_array

from .. import IShapeChanger
from ..pipelines import AbstractPipelineBaseLayer


class BuilderErrorVectorsNorm(IShapeChanger, AbstractPipelineBaseLayer):
    """Compute the error vectors between true and predicted vectors using absolute difference.
    
    This class computes the error vector as the difference between the ground
    truth and the prediction, namely, `norm(gt - pred)`.
    """
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return "BuilderErrorVectorsNorm()"
    
    def __str__(self):
        return "BuilderErrorVectorsNorm"
    
    def __eq__(self, other):
        if not isinstance(other, BuilderErrorVectorsNorm):
            return False
        
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self) -> BuilderErrorVectorsNorm:
        return BuilderErrorVectorsNorm()
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return "n", 1

    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the error vectors using norm over the vector difference.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_elems)
            The predicted vectors.
        
        y : array-like of shape (n_samples, n_elems)
            The target vectors.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        error_vectors : ndarray of shape (n_samples, 1)
            The error vectors computed as `norm(y - x)`.
            
        nothing : ndarray
            An empty array to keep consistency with API.
        """
        check_array(x, force_all_finite="allow-nan")
        check_array(y, force_all_finite="allow-nan")
        gt = np.ma.array(y, mask=np.isnan(y), dtype=np.double)
        pred = np.ma.array(x, mask=np.isnan(x), dtype=np.double)

        errors = gt - pred
        errors = np.square(errors)
        errors = np.sum(errors, axis=1)
        errors = np.sqrt(errors).reshape((-1, 1))
        
        return errors.filled(np.nan), np.array([])