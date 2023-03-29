import abc
from collections.abc import Iterator

import numpy as np


class ICrossValidation(abc.ABC):
    """The interface for cross validation objects.
    """
    
    @abc.abstractmethod
    def get_n_splits(self) -> int:
        """Returns the number of splitting iterations in the cross-validator.
        
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def split(self, x,
              y=None,
              groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices to split data into train and test.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number os samples and
            `n_features` is the number of features.
        
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised problems.
        
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for the split.
        
        test : ndarray
            The testing set indices for the split.
        """
        raise NotImplementedError
