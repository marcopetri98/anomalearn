from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pickle

from sklearn.ensemble import IsolationForest as scikitIsolationForest
import numpy as np

from ....exceptions import InvalidInputShape, NotTrainedError
from ....utils import (are_list_attr_equal, are_normal_attr_equal,
                       are_numpy_attr_equal, are_random_state_attr_equal,
                       are_tree_attr_equal)
from ... import IBoundaryClassifier, IParametric
from ...pipelines import AbstractPipelineSavableLayer
from .. import IAnomalyScorer


class IsolationForest(IParametric, IBoundaryClassifier, IAnomalyScorer, AbstractPipelineSavableLayer):
    """Isolation Forest wrapper for `scikit-learn`.
    
    Attributes
    ----------
    _isolation_forest : scikit-learn IsolationForest
        It is an instance of the scikit-learn `IsolationForest`.
    """
    __scikit_file = "isolation_forest.pickle"

    def __init__(self, n_estimators: int = 100,
                 max_samples: int | float | str = "auto",
                 contamination: float | str = "auto",
                 max_features: int | float = 1.0,
                 bootstrap: bool = False,
                 n_jobs: int | None = None,
                 random_state: int | np.random.RandomState | None = None,
                 verbose: int = 0,
                 warm_start: bool = False):
        super().__init__()
        
        self._isolation_forest = scikitIsolationForest(n_estimators=n_estimators,
                                                       max_samples=max_samples,
                                                       contamination=contamination,
                                                       max_features=max_features,
                                                       bootstrap=bootstrap,
                                                       n_jobs=n_jobs,
                                                       random_state=random_state,
                                                       verbose=verbose,
                                                       warm_start=warm_start)
        
    @property
    def n_estimators(self):
        """Get the `n_estimators` from `scikit` wrapped object.
        
        Returns
        -------
        n_estimators
            The value of `n_estimators`.
        """
        return self._isolation_forest.n_estimators
    
    @n_estimators.setter
    def n_estimators(self, value):
        self._isolation_forest.n_estimators = value
        
    @property
    def max_samples(self):
        """Get the `max_samples` from `scikit` wrapped object.
        
        Returns
        -------
        max_samples
            The value of `max_samples`.
        """
        return self._isolation_forest.max_samples
    
    @max_samples.setter
    def max_samples(self, value):
        self._isolation_forest.max_samples = value
        
    @property
    def contamination(self):
        """Get the `contamination` from `scikit` wrapped object.
        
        Returns
        -------
        contamination
            The value of `contamination`.
        """
        return self._isolation_forest.contamination
    
    @contamination.setter
    def contamination(self, value):
        self._isolation_forest.contamination = value
        
    @property
    def max_features(self):
        """Get the `max_features` from `scikit` wrapped object.
        
        Returns
        -------
        max_features
            The value of `max_features`.
        """
        return self._isolation_forest.max_features
    
    @max_features.setter
    def max_features(self, value):
        self._isolation_forest.max_features = value
        
    @property
    def bootstrap(self):
        """Get the `bootstrap` from `scikit` wrapped object.
        
        Returns
        -------
        bootstrap
            The value of `bootstrap`.
        """
        return self._isolation_forest.bootstrap
    
    @bootstrap.setter
    def bootstrap(self, value):
        self._isolation_forest.bootstrap = value
        
    @property
    def n_jobs(self):
        """Get the `n_jobs` from `scikit` wrapped object.
        
        Returns
        -------
        n_jobs
            The value of `n_jobs`.
        """
        return self._isolation_forest.n_jobs
    
    @n_jobs.setter
    def n_jobs(self, value):
        self._isolation_forest.n_jobs = value
        
    @property
    def random_state(self):
        """Get the `random_state` from `scikit` wrapped object.
        
        Returns
        -------
        random_state
            The value of `random_state`.
        """
        return self._isolation_forest.random_state
    
    @random_state.setter
    def random_state(self, value):
        self._isolation_forest.random_state = value
        
    @property
    def verbose(self):
        """Get the `verbose` from `scikit` wrapped object.
        
        Returns
        -------
        verbose
            The value of `verbose`.
        """
        return self._isolation_forest.verbose
    
    @verbose.setter
    def verbose(self, value):
        self._isolation_forest.verbose = value
        
    @property
    def warm_start(self):
        """Get the `warm_start` from `scikit` wrapped object.
        
        Returns
        -------
        warm_start
            The value of `warm_start`.
        """
        return self._isolation_forest.warm_start
    
    @warm_start.setter
    def warm_start(self, value):
        self._isolation_forest.warm_start = value
        
    @property
    def base_estimator(self):
        """Get the `base_estimator_` from `scikit` wrapped object if present.
        
        Returns
        -------
        base_estimator_
            The value of `base_estimator_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.base_estimator_
        except AttributeError:
            return None
        
    @property
    def estimators(self):
        """Get the `estimators_` from `scikit` wrapped object if present.
        
        Returns
        -------
        estimators_
            The value of `estimators_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.estimators_
        except AttributeError:
            return None
        
    @property
    def estimators_features(self):
        """Get the `estimators_features_` from `scikit` wrapped object if present.
        
        Returns
        -------
        estimators_features_
            The value of `estimators_features_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.estimators_features_
        except AttributeError:
            return None
        
    @property
    def estimators_samples(self):
        """Get the `estimators_samples_` from `scikit` wrapped object if present.
        
        Returns
        -------
        estimators_samples_
            The value of `estimators_samples_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.estimators_samples_
        except AttributeError:
            return None
        
    @property
    def seen_max_samples(self):
        """Get the `max_samples_` from `scikit` wrapped object if present.
        
        Returns
        -------
        max_samples_
            The value of `max_samples_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.max_samples_
        except AttributeError:
            return None
        
    @property
    def offset(self):
        """Get the `offset_` from `scikit` wrapped object if present.
        
        Returns
        -------
        offset_
            The value of `offset_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.offset_
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
            return self._isolation_forest.n_features_in_
        except AttributeError:
            return None
        
    @property
    def seen_feature_names_in(self):
        """Get the `feature_names_in_` from `scikit` wrapped object if present.
        
        Returns
        -------
        feature_names_in_
            The value of `feature_names_in_` if present, None otherwise.
        """
        try:
            return self._isolation_forest.feature_names_in_
        except AttributeError:
            return None

    def __repr__(self):
        return f"IsolationForest(n_estimators={self.n_estimators}, " \
               f"max_samples={self.max_samples}, " \
               f"contamination={self.contamination}, " \
               f"max_features={self.max_features}, " \
               f"bootstrap={self.bootstrap}, " \
               f"n_jobs={self.n_jobs}, " \
               f"random_state={self.random_state}, " \
               f"verbose={self.verbose}, " \
               f"warm_start={self.warm_start})"

    def __str__(self):
        return "IsolationForest"

    def __eq__(self, other):
        if not isinstance(other, IsolationForest):
            return False

        numpy_properties = ["seen_feature_names_in"]
        normal_properties = ["n_estimators", "max_samples", "contamination",
                             "max_features", "bootstrap", "n_jobs", "verbose",
                             "warm_start", "seen_max_samples", "offset",
                             "seen_features_in"]
        list_properties = ["estimators_features", "estimators_samples"]
        random_states = ["random_state"]
        if not are_numpy_attr_equal(self, other, numpy_properties) or \
                not are_normal_attr_equal(self, other, normal_properties) or \
                not are_list_attr_equal(self, other, list_properties) or \
                not are_random_state_attr_equal(self, other, random_states):
            return False

        # check that the contained extra tree regressors are identical
        xtr_normal = ["criterion", "splitter", "max_depth", "min_samples_split",
                      "min_samples_leaf", "min_weight_fraction_leaf",
                      "max_features", "min_impurity_decrease", "max_leaf_nodes",
                      "ccp_alpha", "max_features_", "n_features_in_", "n_outputs_"]
        xtr_numpy = ["feature_names_in_", "feature_importances_"]
        xtr_random_state = ["random_state"]
        xtr_tree = ["tree_"]

        def are_equal_extra_tree_regressor(xtr1, xtr2) -> bool:
            if (xtr1 is None) != (xtr2 is None):
                return False
            
            if xtr1 is not None:
                if not are_normal_attr_equal(xtr1, xtr2, xtr_normal) or \
                        not are_numpy_attr_equal(xtr1, xtr2, xtr_numpy) or \
                        not are_random_state_attr_equal(xtr1, xtr2, xtr_random_state) or \
                        not are_tree_attr_equal(xtr1, xtr2, xtr_tree):
                    return False
            
            return True
        
        if not are_equal_extra_tree_regressor(self.base_estimator, other.base_estimator):
            return False
        
        # check that the list of extra tree regressors is identical
        if (self.estimators is None) != (other.estimators is None):
            return False
        
        if self.estimators is not None:
            if len(self.estimators) != len(other.estimators):
                return False
            
            for tree1, tree2 in zip(self.estimators, other.estimators):
                if not are_equal_extra_tree_regressor(tree1, tree2):
                    return False
        
        return True
    
    def copy(self) -> IsolationForest:
        """Copies the object.
        
        Note that since scikit-learn does not provide standard `save` and `load`
        methods for objects, and it does not provide a complete copy method,
        deepcopy will be used.
        
        Returns
        -------
        new_object : IsolationForest
            The copied object.
        """
        new = IsolationForest(n_estimators=self.n_estimators,
                              max_samples=self.max_samples,
                              contamination=self.contamination,
                              max_features=self.max_features,
                              bootstrap=self.bootstrap,
                              n_jobs=self.n_jobs,
                              random_state=self.random_state,
                              verbose=self.verbose,
                              warm_start=self.warm_start)
        new._isolation_forest = deepcopy(self._isolation_forest)
        return new
    
    def save(self, path,
             *args,
             **kwargs) -> IsolationForest:
        super().save(path=path, *args, **kwargs)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "wb") as f:
            pickle.dump(self._isolation_forest, f)
            
        return self
    
    def load(self, path: str,
             *args,
             **kwargs) -> IsolationForest:
        super().load(path=path, *args, **kwargs)
        path_obj = Path(path)
        
        with open(str(path_obj / self.__scikit_file), "rb") as f:
            self._isolation_forest = pickle.load(f)
            
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> IsolationForest:
        obj = IsolationForest()
        obj.load(path, *args, **kwargs)
        return obj
        
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return tuple(["n"])
        
    def fit(self, x, y=None, *args, **kwargs) -> None:
        self._isolation_forest.fit(x, y)
    
    def classify(self, x, *args, **kwargs) -> np.ndarray:
        if self.seen_features_in is None:
            raise NotTrainedError()

        x = np.array(x)
        
        if x.shape[1] != self.seen_features_in:
            raise InvalidInputShape(("n_points", self.seen_features_in), x.shape)
        
        return self._isolation_forest.predict(x)
    
    def decision_function(self, x, *args, **kwargs) -> np.ndarray:
        """Opposite of scikit-learn `decision_function`.
        """
        if self.seen_features_in is None:
            raise NotTrainedError()

        x = np.array(x)
        
        if x.shape[1] != self.seen_features_in:
            raise InvalidInputShape(("n_points", self.seen_features_in), x.shape)
        
        return self._isolation_forest.decision_function(x) * -1
    
    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        """Opposite of scikit-learn `score_samples`.
        """
        if self.seen_features_in is None:
            raise NotTrainedError()

        x = np.array(x)
        
        if x.shape[1] != self.seen_features_in:
            raise InvalidInputShape(("n_points", self.seen_features_in), x.shape)
        
        return self._isolation_forest.score_samples(x) * -1
