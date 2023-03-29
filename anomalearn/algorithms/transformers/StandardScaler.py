from __future__ import annotations

import pickle
from copy import deepcopy
from numbers import Number
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler as scikitStandardScaler

from .. import ITransformer, IParametric
from ..pipelines import AbstractPipelineSavableLayer
from ...exceptions import InvalidInputShape, NotTrainedError
from ...utils import are_numpy_attr_equal, are_normal_attr_equal


class StandardScaler(ITransformer, IParametric, AbstractPipelineSavableLayer):
    """Standard scaler wrapper for `scikit-learn`.
    
    Attributes
    ----------
    _standard_scaler : scikit-learn StandardScaler
        It is an instance of the scikit-learn `StandardScaler`.
    """
    __scikit_file = "standard_scaler.pickle"
    
    def __init__(self, copy: bool = True,
                 with_mean: bool = True,
                 with_std: bool = True):
        super().__init__()
        
        self._standard_scaler = scikitStandardScaler(copy=copy,
                                                     with_mean=with_mean,
                                                     with_std=with_std)
        
    @property
    def copy_attribute(self):
        return self._standard_scaler.copy
    
    @copy_attribute.setter
    def copy_attribute(self, value):
        self._standard_scaler.copy = value
        
    @property
    def with_mean(self):
        return self._standard_scaler.with_mean
    
    @with_mean.setter
    def with_mean(self, value):
        self._standard_scaler.with_mean = value
        
    @property
    def with_std(self):
        return self._standard_scaler.with_std
    
    @with_std.setter
    def with_std(self, value):
        self._standard_scaler.with_std = value
        
    @property
    def seen_scale(self):
        try:
            return self._standard_scaler.scale_
        except AttributeError:
            return None
        
    @property
    def seen_mean(self):
        try:
            return self._standard_scaler.mean_
        except AttributeError:
            return None
        
    @property
    def seen_var(self):
        try:
            return self._standard_scaler.var_
        except AttributeError:
            return None
    
    @property
    def seen_features_in(self):
        try:
            return self._standard_scaler.n_features_in_
        except AttributeError:
            return None
    
    @property
    def seen_samples_in(self):
        try:
            return self._standard_scaler.n_samples_seen_
        except AttributeError:
            return None
    
    @property
    def seen_features_names_in(self):
        try:
            return self._standard_scaler.feature_names_in_
        except AttributeError:
            return None
        
    def __repr__(self):
        return f"StandardScaler(copy={self.copy_attribute},with_mean={self.with_mean},with_std={self.with_std})"
    
    def __str__(self):
        return "StandardScaler"
    
    def __eq__(self, other):
        if not isinstance(other, StandardScaler):
            return False
        
        numpy_properties = ["seen_scale", "seen_mean", "seen_var", "seen_features_names_in"]
        normal_properties = ["copy_attribute", "with_mean", "with_std", "seen_features_in"]
        if not are_numpy_attr_equal(self, other, numpy_properties):
            return False
        if not are_normal_attr_equal(self, other, normal_properties):
            return False

        if (self.seen_samples_in is None) != (self.seen_samples_in is None):
            return False
        
        if isinstance(self.seen_samples_in, np.ndarray):
            if self.seen_samples_in is not None and not np.array_equal(self.seen_samples_in, other.seen_samples_in):
                return False
        elif isinstance(self.seen_samples_in, Number):
            if self.seen_samples_in is not None and self.seen_samples_in != other.seen_samples_in:
                return False
        
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> StandardScaler:
        """Copies the object.
        
        Note that since scikit-learn does not provide standard `save` and `load`
        methods for objects, and it does not provide a complete copy method,
        deepcopy will be used.
        
        Returns
        -------
        new_object : MinMaxScaler
            The copied object.
        """
        new = StandardScaler(self.copy_attribute, self.with_mean, self.with_std)
        new._standard_scaler = deepcopy(self._standard_scaler)
        return new
    
    def save(self, path,
             *args,
             **kwargs) -> StandardScaler:
        super().save(path=path)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "wb") as f:
            pickle.dump(self._standard_scaler, f)
        
        return self
    
    def load(self, path: str,
             *args,
             **kwargs) -> StandardScaler:
        super().load(path=path)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "rb") as f:
            self._standard_scaler = pickle.load(f)
        
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> StandardScaler:
        obj = StandardScaler()
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return "n", "m"
    
    def fit(self, x, y=None, *args, **kwargs) -> None:
        self._standard_scaler.fit(x)
    
    def transform(self, x, *args, **kwargs) -> np.ndarray:
        if self.seen_mean is None:
            raise NotTrainedError()
        
        if x.shape[1] != self.seen_mean.shape[0]:
            raise InvalidInputShape(("n_points", self.seen_mean.shape[0]), x.shape)
        
        return self._standard_scaler.transform(x)
