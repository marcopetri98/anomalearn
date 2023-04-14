from __future__ import annotations

from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Tuple
import pickle

from sklearn.preprocessing import MinMaxScaler as scikitMinMaxScaler
import numpy as np

from ...exceptions import InvalidInputShape, NotTrainedError
from ...utils import are_normal_attr_equal, are_numpy_attr_equal
from .. import IParametric, ITransformer
from ..pipelines import AbstractPipelineSavableLayer


class MinMaxScaler(ITransformer, IParametric, AbstractPipelineSavableLayer):
    """Min max scaler wrapper for `scikit-learn`.
    
    Attributes
    ----------
    _min_max_scaler : scikit-learn MinMaxScaler
        It is an instance of the scikit-learn `MinMaxScaler`.
    """
    __scikit_file = "min_max_scaler.pickle"
    
    def __init__(self, feature_range: Tuple[Number, Number] = (0, 1),
                 copy: bool = True,
                 clip: bool = False):
        super().__init__()
        
        self._min_max_scaler = scikitMinMaxScaler(feature_range=feature_range,
                                                  copy=copy,
                                                  clip=clip)
    
    @property
    def feature_range(self):
        """Get the `feature_range` from `scikit` wrapped object.
        
        Returns
        -------
        feature_range
            The value of `feature_range`.
        """
        return self._min_max_scaler.feature_range
    
    @feature_range.setter
    def feature_range(self, value):
        self._min_max_scaler.feature_range = value
        
    @property
    def copy_attribute(self):
        """Get the `copy` from `scikit` wrapped object.
        
        Returns
        -------
        copy
            The value of `copy`.
        """
        return self._min_max_scaler.copy
    
    @copy_attribute.setter
    def copy_attribute(self, value):
        self._min_max_scaler.copy = value
        
    @property
    def clip(self):
        """Get the `clip` from `scikit` wrapped object.
        
        Returns
        -------
        clip
            The value of `clip`.
        """
        return self._min_max_scaler.clip
    
    @clip.setter
    def clip(self, value):
        self._min_max_scaler.clip = value
    
    @property
    def scale_adjustment(self):
        """Get the `scale_` from `scikit` wrapped object if present.
        
        Returns
        -------
        scale_
            The value of `scale_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.scale_
        except AttributeError:
            return None
    
    @property
    def min_adjustment(self):
        """Get the `min_` from `scikit` wrapped object if present.
        
        Returns
        -------
        min_
            The value of `min_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.min_
        except AttributeError:
            return None
        
    @property
    def seen_data_min(self):
        """Get the `data_min_` from `scikit` wrapped object if present.
        
        Returns
        -------
        data_min_
            The value of `data_min_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.data_min_
        except AttributeError:
            return None
        
    @property
    def seen_data_max(self):
        """Get the `data_max_` from `scikit` wrapped object if present.
        
        Returns
        -------
        data_max_
            The value of `data_max_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.data_max_
        except AttributeError:
            return None
    
    @property
    def seen_data_range(self):
        """Get the `data_range_` from `scikit` wrapped object if present.
        
        Returns
        -------
        data_range_
            The value of `data_range_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.data_range_
        except AttributeError:
            return None
    
    @property
    def seen_features_in(self):
        """Get the `n_features_in_` from `scikit` wrapped object if present.
        
        Returns
        -------
        n_features_in_
            The value of `n_features_in_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.n_features_in_
        except AttributeError:
            return None
    
    @property
    def seen_samples_in(self):
        """Get the `n_samples_seen_` from `scikit` wrapped object if present.
        
        Returns
        -------
        n_samples_seen_
            The value of `n_samples_seen_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.n_samples_seen_
        except AttributeError:
            return None
    
    @property
    def seen_features_names_in(self):
        """Get the `feature_names_in_` from `scikit` wrapped object if present.
        
        Returns
        -------
        feature_names_in_
            The value of `feature_names_in_` if present, None otherwise.
        """
        try:
            return self._min_max_scaler.feature_names_in_
        except AttributeError:
            return None
        
    def __repr__(self):
        return f"MinMaxScaler(feature_range={self.feature_range}, copy={self.copy_attribute}, clip={self.clip})"
    
    def __str__(self):
        return "MinMaxScaler"
    
    def __eq__(self, other):
        if not isinstance(other, MinMaxScaler):
            return False
        
        numpy_properties = ["scale_adjustment", "min_adjustment", "seen_data_min", "seen_data_max", "seen_data_range", "seen_features_names_in"]
        normal_properties = ["feature_range", "copy_attribute", "clip", "seen_features_in", "seen_samples_in"]
        if not are_numpy_attr_equal(self, other, numpy_properties):
            return False
        if not are_normal_attr_equal(self, other, normal_properties):
            return False
        
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> MinMaxScaler:
        """Copies the object.
        
        Note that since scikit-learn does not provide standard `save` and `load`
        methods for objects, and it does not provide a complete copy method,
        deepcopy will be used.
        
        Returns
        -------
        new_object : MinMaxScaler
            The copied object.
        """
        new = MinMaxScaler(self.feature_range, self.copy_attribute, self.clip)
        new._min_max_scaler = deepcopy(self._min_max_scaler)
        return new
        
    def save(self, path,
             *args,
             **kwargs) -> MinMaxScaler:
        super().save(path=path)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "wb") as f:
            pickle.dump(self._min_max_scaler, f)
            
        return self
    
    def load(self, path: str,
             *args,
             **kwargs) -> MinMaxScaler:
        super().load(path=path)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "rb") as f:
            self._min_max_scaler = pickle.load(f)
            
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> MinMaxScaler:
        obj = MinMaxScaler()
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return "n", "m"
        
    def fit(self, x, y=None, *args, **kwargs) -> None:
        self._min_max_scaler.fit(x)
        
    def transform(self, x, *args, **kwargs) -> np.ndarray:
        if self.seen_data_max is None:
            raise NotTrainedError()
            
        x = np.array(x)
        
        if x.shape[1] != self.seen_data_max.shape[0]:
            raise InvalidInputShape(("n_points", self.seen_data_max.shape[0]), x.shape)
        
        return self._min_max_scaler.transform(x)
