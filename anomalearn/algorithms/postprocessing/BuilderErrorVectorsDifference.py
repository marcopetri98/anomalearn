from __future__ import annotations

from typing import Tuple

from sklearn.utils import check_array
import numpy as np

from .. import IShapeChanger
from ..pipelines import AbstractPipelineBaseLayer


class BuilderErrorVectorsDifference(IShapeChanger, AbstractPipelineBaseLayer):
    """Compute the error vectors between true and predicted vectors using difference.
    
    This class computes the error vector as the difference between the ground
    truth and the prediction, namely, `gt - pred`.
    """
        
    def __repr__(self):
        return "BuilderErrorVectorsDifference()"
    
    def __str__(self):
        return "BuilderErrorVectorsDifference"
    
    def __eq__(self, other):
        if not isinstance(other, BuilderErrorVectorsDifference):
            return False
        
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def copy(self) -> BuilderErrorVectorsDifference:
        return BuilderErrorVectorsDifference()
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return "n", "m"
        
    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the error vectors using difference.
        
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
        error_vectors : ndarray of shape (n_samples, n_elems)
            The error vectors computed as `y - x`.
            
        nothing : ndarray
            An empty array to keep consistency with API.
        """
        check_array(x, force_all_finite="allow-nan")
        check_array(y, force_all_finite="allow-nan")
        ground_truth = np.ma.array(y, mask=np.isnan(y), dtype=np.double)
        pred = np.ma.array(x, mask=np.isnan(x), dtype=np.double)
        
        errors = ground_truth - pred
        
        return errors.filled(np.nan), np.array([])
