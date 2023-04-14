from __future__ import annotations

from pathlib import Path
from typing import Tuple
import os

from scipy.spatial.distance import mahalanobis
from sklearn.utils import check_array
import numpy as np

from ...exceptions import NotTrainedError
from ...utils import estimate_mean_covariance, get_rows_without_nan
from .. import IParametric, IShapeChanger
from ..pipelines import AbstractPipelineSavableLayer


class ScorerMahalanobis(IShapeChanger, IParametric, AbstractPipelineSavableLayer):
    """Score computer using mahalanobis distance based on errors.
    
    As proposed by Malhotra et al. (https://sites.google.com/site/icmlworkshoponanomalydetection/accepted-papers),
    the score of a time point is computed as the mahalanobis distance between
    the error vector and the mean vector of the estimated multivariate gaussian
    distribution on the errors.
    
    Attributes
    ----------
    _mean : ndarray
        It is the mean vector that will be estimated by the `fit` method.
    
    _cov : ndarray
        It is the covariance matrix or the standard deviation that will be
        estimated by the `fit` method.
    
    _inv_cov : ndarray
        The inverse of the covariance matrix.
    """
    __numpy_file = "score_mahalanobis.npz"
    
    def __init__(self):
        super().__init__()
        
        self._mean = None
        self._cov = None
        self._inv_cov = None
        
    @property
    def mean(self):
        """Gets the computed mean.
        
        Returns
        -------
        mean
            The computed mean or None if not yet computed.
        """
        return self._mean
        
    @property
    def cov(self):
        """Gets the computed covariance matrix or standard deviation.
        
        Returns
        -------
        cov_or_std
            The computed covariance matrix or standard deviation, or None if
            not yet computed.
        """
        return self._cov
        
    @property
    def inv_cov(self):
        """Gets the computed inverse of the covariance matrix or standard deviation.
        
        Returns
        -------
        inv_cov_or_std
            The computed inverse of the covariance matrix or standard deviation,
            or None if not yet computed.
        """
        return self._inv_cov
        
    def __repr__(self):
        return "ScorerMahalanobis()"
    
    def __str__(self):
        return "ScorerMahalanobis"
    
    def __eq__(self, other):
        if not isinstance(other, ScorerMahalanobis):
            return False
        
        # if the mean is None, the fit has not been called and all are None
        if self._mean is None and other._mean is None and self._cov is None and \
                other._cov is None and self._inv_cov is None and other._inv_cov is None:
            return True
        elif self._mean is None and other._mean is not None or \
                self._mean is not None and other._mean is None:
            return False
        elif self._cov is None and other._cov is not None or \
                self._cov is not None and other._cov is None:
            return False
        elif self._inv_cov is None and other._inv_cov is not None or \
                self._inv_cov is not None and other._inv_cov is None:
            return False
        else:
            mean_eq = np.array_equal(self._mean, other._mean, equal_nan=True)
            cov_eq = np.array_equal(self._cov, other._cov, equal_nan=True)
            inv_cov_eq = np.array_equal(self._inv_cov, other._inv_cov, equal_nan=True)
            return mean_eq and cov_eq and inv_cov_eq
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> ScorerMahalanobis:
        new = ScorerMahalanobis()
        new._mean = self._mean.copy() if self._mean is not None else None
        new._cov = self._cov.copy() if self._cov is not None else None
        new._inv_cov = self._inv_cov.copy() if self._inv_cov is not None else None
        return new
        
    def save(self, path,
             *args,
             **kwargs) -> ScorerMahalanobis:
        super().save(path=path)
        path_obj = Path(path)

        if self._mean is not None:
            np.savez_compressed(str(path_obj / self.__numpy_file),
                                _mean=self._mean,
                                _cov=self._cov,
                                _inv_cov=self._inv_cov)
        
        return self

    def load(self, path: str,
             *args,
             **kwargs) -> ScorerMahalanobis:
        super().load(path=path)
        path_obj = Path(path)

        if not path_obj.is_dir():
            raise ValueError("path must be a directory")

        if self.__numpy_file in os.listdir(str(path_obj)):
            with np.load(str(path_obj / self.__numpy_file)) as data:
                self._mean = data["_mean"]
                self._cov = data["_cov"]
                self._inv_cov = data["_inv_cov"]
        else:
            self._mean = None
            self._cov = None
            self._inv_cov = None
        
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> ScorerMahalanobis:
        obj = ScorerMahalanobis()
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return tuple(["n"])
    
    def fit(self, x, y=None, *args, **kwargs) -> None:
        """Learns the mean vector and the covariance matrix.

        The mean and the covariance matrix are computed from the errors
        contained in `x`. For efficiency reasons, the inverse of the covariance
        matrix is kept in memory such that consecutive shape changes won't
        involve the computation of a matrix inverse. The mean vector and the
        covariance matrix are estimated using `estimate_mean_covariance`.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_elems)
            It is the array containing the prediction/reconstruction errors of
            the model.

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
        self._mean, self._cov, self._inv_cov = estimate_mean_covariance(x)
    
    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the anomaly scores using mahalanobis distance.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_elems)
            It is the array containing the prediction/reconstruction errors of
            the model.

        y
            Ignored.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        anomaly_scores : ndarray of shape (n_samples,)
            The anomaly scores of points, the higher the score the more likely
            the point is anomalous.

        nothing : ndarray
            Empty numpy array to respect API.
        """
        check_array(x, force_all_finite="allow-nan")
        
        if self._mean is None:
            raise NotTrainedError()
        
        rows_without_nan = get_rows_without_nan(x)
        
        scores = np.full(x.shape[0], fill_value=np.nan)
        clean_errors = x[rows_without_nan]

        valid_scores = np.array([mahalanobis(clean_errors[i], self._mean, self._inv_cov)
                                 for i in range(clean_errors.shape[0])])
        scores[rows_without_nan] = valid_scores
        
        return scores, np.array([])
