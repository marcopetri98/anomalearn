from __future__ import annotations

from pathlib import Path

import numpy as np

from ...exceptions import NotTrainedError
from ...input_validation import check_array_1d
from ...utils import load_py_json, save_py_json
from .. import IParametric, ITransformer
from ..pipelines import AbstractPipelineSavableLayer


class ThresholdMaxOnNormal(IParametric, ITransformer, AbstractPipelineSavableLayer):
    """Computes the threshold as the maximum anomaly score considering the input as scores of normal points.
    
    Scores are assumed to be higher for points considered more anomalous. Thus,
    the higher the score the more anomalous the time point.
    """
    __json_file = "threshold_max_on_normal.json"
    
    def __init__(self):
        super().__init__()
        
        self._threshold = None
        
    @property
    def threshold(self):
        """Gets the computed threshold.
        
        Returns
        -------
        threshold
            The computed threshold or None if not yet computed.
        """
        return self._threshold
        
    def __repr__(self):
        return "ThresholdMaxOnNormal()"
    
    def __str__(self):
        return "ThresholdMaxOnNormal"
    
    def __eq__(self, other):
        if not isinstance(other, ThresholdMaxOnNormal):
            return False
        
        if self._threshold is None and other._threshold is None:
            return True
        else:
            return self._threshold == other._threshold
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> ThresholdMaxOnNormal:
        new = ThresholdMaxOnNormal()
        new._threshold = self._threshold
        return new
    
    def save(self, path,
             *args,
             **kwargs) -> ThresholdMaxOnNormal:
        super().save(path=path)
        path_obj = Path(path)
        
        save_py_json(self._threshold, str(path_obj / self.__json_file))
        return self
    
    def load(self, path: str,
             *args,
             **kwargs) -> ThresholdMaxOnNormal:
        super().load(path=path)
        path_obj = Path(path)
        
        self._threshold = load_py_json(str(path_obj / self.__json_file))
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> ThresholdMaxOnNormal:
        obj = ThresholdMaxOnNormal()
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        return tuple(["n"])
    
    def get_output_shape(self) -> tuple:
        return tuple(["n"])
        
    def fit(self, x, y=None, *args, **kwargs) -> None:
        """Learns the threshold from an array of anomaly scores.
        
        Parameters
        ----------
        x : array-like of shape (n_points,)
            An array like containing anomaly scores.
        
        y
            Ignored.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        check_array_1d(x, "allow-nan")
        scores = np.ma.array(x, mask=np.isnan(x))
        self._threshold = np.ma.max(scores)
    
    def transform(self, x, *args, **kwargs) -> np.ndarray:
        """Extracts labels from the anomaly scores by comparing them with the threshold.
        
        Parameters
        ----------
        x : array-like of shape (n_points,)
            The anomaly scores of the points.

        Returns
        -------
        labels : ndarray of shape (n_points,)
            The labels for the points in input.
            
        Raises
        ------
        ValueError
            If the transform is called before the fit has been called.
        """
        check_array_1d(x, "allow-nan")
        x = np.ma.array(x, mask=np.isnan(x))
        
        if self._threshold is None:
            raise NotTrainedError()
        
        return (x > self._threshold).filled(np.nan)
