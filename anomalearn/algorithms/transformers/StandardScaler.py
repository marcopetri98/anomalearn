from __future__ import annotations

from copy import deepcopy
from numbers import Number
from pathlib import Path
import pickle

from sklearn.preprocessing import StandardScaler as scikitStandardScaler
import numpy as np

from ...exceptions import InvalidInputShape, NotTrainedError
from ...utils import are_normal_attr_equal, are_numpy_attr_equal
from .. import IParametric, ITransformer
from ..pipelines import AbstractPipelineSavableLayer


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
        """Get the `copy` from `scikit` wrapped object.
        
        Returns
        -------
        copy
            The value of `copy`.
        """
        return self._standard_scaler.copy
    
    @copy_attribute.setter
    def copy_attribute(self, value):
        self._standard_scaler.copy = value
        
    @property
    def with_mean(self):
        """Get the `with_mean` from `scikit` wrapped object.
        
        Returns
        -------
        with_mean
            The value of `with_mean`.
        """
        return self._standard_scaler.with_mean
    
    @with_mean.setter
    def with_mean(self, value):
        self._standard_scaler.with_mean = value
        
    @property
    def with_std(self):
        """Get the `with_std` from `scikit` wrapped object.
        
        Returns
        -------
        with_std
            The value of `with_std`.
        """
        return self._standard_scaler.with_std
    
    @with_std.setter
    def with_std(self, value):
        self._standard_scaler.with_std = value
        
    @property
    def seen_scale(self):
        """Get the `scale_` from `scikit` wrapped object if present.
        
        Returns
        -------
        scale_
            The value of `scale_` if present, None otherwise.
        """
        try:
            return self._standard_scaler.scale_
        except AttributeError:
            return None
        
    @property
    def seen_mean(self):
        """Get the `mean_` from `scikit` wrapped object if present.
        
        Returns
        -------
        mean_
            The value of `mean_` if present, None otherwise.
        """
        try:
            return self._standard_scaler.mean_
        except AttributeError:
            return None
        
    @property
    def seen_var(self):
        """Get the `var_` from `scikit` wrapped object if present.
        
        Returns
        -------
        var_
            The value of `var_` if present, None otherwise.
        """
        try:
            return self._standard_scaler.var_
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
            return self._standard_scaler.n_features_in_
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
            return self._standard_scaler.n_samples_seen_
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
        if not are_numpy_attr_equal(self, other, numpy_properties) or \
                not are_normal_attr_equal(self, other, normal_properties):
            return False

        if (self.seen_samples_in is None) != (self.seen_samples_in is None):
            return False
        elif self.seen_samples_in is not None and not isinstance(self.seen_samples_in, other.seen_samples_in.__class__):
            return False
        
        if isinstance(self.seen_samples_in, np.ndarray) and isinstance(other.seen_samples_in, np.ndarray) and \
                not np.array_equal(self.seen_samples_in, other.seen_samples_in):
            return False
        elif isinstance(self.seen_samples_in, Number) and isinstance(other.seen_samples_in, Number) and \
                self.seen_samples_in != other.seen_samples_in:
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
