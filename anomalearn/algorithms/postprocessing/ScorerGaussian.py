from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils import check_array

from .. import IShapeChanger, IParametric
from ..pipelines import AbstractPipelineSavableLayer
from ...exceptions import NotTrainedError
from ...utils import estimate_mean_covariance
from ...utils import get_rows_without_nan


class ScorerGaussian(IShapeChanger, IParametric, AbstractPipelineSavableLayer):
    """Score computer using multivariate gaussian distribution based on errors.

    As proposed by Malhotra et al. (https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf),
    the score of a point is computed starting from the probability density
    function of a multivariate gaussian distribution. However, since in this
    approach the lower the pdf the higher the anomaly, the score will be
    computed as the inverse of the pdf such that the higher the score the higher
    the anomaly.
    
    Attributes
    ----------
    _mean : ndarray
        It is the mean vector that will be estimated by the `fit` method.
    
    _cov : ndarray
        It is the covariance matrix or the standard deviation that will be
        estimated by the `fit` method.
    """
    __numpy_file = "scorer_gaussian.npz"

    def __init__(self):
        super().__init__()

        self._mean = None
        self._cov = None
        
    @property
    def mean(self):
        return self._mean
        
    @property
    def cov(self):
        return self._cov
        
    def __repr__(self):
        return "ScorerGaussian()"
    
    def __str__(self):
        return "ScorerGaussian"
    
    def __eq__(self, other):
        if not isinstance(other, ScorerGaussian):
            return False

        # if the mean is None also cov is None
        if self._mean is None and other._mean is None and self._cov is None and other._cov is None:
            return True
        elif self._mean is None and other._mean is not None or self._mean is not None and other._mean is None:
            return False
        elif self._cov is None and other._cov is not None or self._cov is not None and other._cov is None:
            return False
        else:
            mean_eq = np.array_equal(self._mean, other._mean, equal_nan=True)
            cov_eq = np.array_equal(self._cov, other._cov, equal_nan=True)
            return mean_eq and cov_eq
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> ScorerGaussian:
        new = ScorerGaussian()
        new._mean = self._mean.copy() if self._mean is not None else None
        new._cov = self._cov.copy() if self._cov is not None else None
        return new

    def save(self, path,
             *args,
             **kwargs) -> ScorerGaussian:
        super().save(path=path)
        path_obj = Path(path)

        if self._mean is not None:
            np.savez_compressed(str(path_obj / self.__numpy_file),
                                _mean=self._mean,
                                _cov=self._cov)
        return self

    def load(self, path: str,
             *args,
             **kwargs) -> ScorerGaussian:
        super().load(path=path)
        path_obj = Path(path)

        if not path_obj.is_dir():
            raise ValueError("path must be a directory")

        if self.__numpy_file in os.listdir(str(path_obj)):
            with np.load(str(path_obj / self.__numpy_file)) as data:
                self._mean = data["_mean"]
                self._cov = data["_cov"]
        else:
            self._mean = None
            self._cov = None
        
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> ScorerGaussian:
        obj = ScorerGaussian()
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return tuple(["n"])

    def fit(self, x, y=None, *args, **kwargs) -> None:
        """Learns the mean vector and the covariance matrix.

        The mean and the covariance matrix are computed from the errors
        contained in `x`. The mean vector and the covariance matrix are
        estimated using `estimate_mean_covariance`.

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
        self._mean, self._cov, _ = estimate_mean_covariance(x)

    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the scores for the points.
        
        The function translate the vectors into scores. Since to compute the
        probability it is needed a complete vector, if a vector contains at
        least one NaN, its score will be NaN.

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
            the point is anomalous. To the value of the pdf it is added the
            quantity 1e-10 before computing its inverse to avoid division by
            zero.

        nothing : ndarray
            Empty numpy array to respect API.
        """
        check_array(x, force_all_finite="allow-nan")
        
        if self._mean is None:
            raise NotTrainedError()
        
        rows_without_nan = get_rows_without_nan(x)
        
        scores = np.full(x.shape[0], fill_value=np.nan)
        clean_errors = x[rows_without_nan]

        proba = multivariate_normal.pdf(clean_errors,
                                        mean=self._mean,
                                        cov=self._cov,
                                        allow_singular=True) + 1e-10
        valid_scores = 1 / proba
        scores[rows_without_nan] = valid_scores
        
        return scores, np.array([])
