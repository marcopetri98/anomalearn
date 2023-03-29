import os
from abc import ABC
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.linalg import LinAlgError
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
from sklearn.utils import check_array

from .... import IParametric, SavableModel
from .....input_validation import check_argument_types
from .....utils import print_step, print_warning, save_py_json, find_or_create_dir, load_py_json


class TSAErrorBased(IParametric, SavableModel, ABC):
    """Abstract class for models performing anomaly detection based on errors

    This class describes a type of semi-supervised learning framework for
    anomaly detection based on errors. The errors could be prediction errors as
    well as reconstruction errors. Therefore, the class implements some general
    methods to compute the errors (given the ground truth, i.e., the real values
    to predict or reconstruct, and the predictions/reconstructions [note: gt is
    not referred to the labels here]) and the threshold. All of these methods
    define a way to compute an anomaly score (the error) and a threshold to
    decide if a score identifies an anomaly or a normal point.
    
    Parameters
    ----------
    error_method : ["difference", "abs_difference", "norm", "custom"], default="difference"
        It is the way in which the error is computed between the gt and the
        predictions (remember, with gt we meant the value to be predicted or
        reconstructed, not the anomaly labels). Here there is a description:
        
        - **difference**: compute the difference between the gt and the
          prediction/reconstruction.
        - **abs_difference**: absolute value of difference (see first point)
        - **norm**: computes the norm of the difference vector between the gt and
          the prediction/reconstruction.
        - **custom**: custom function specified by the user.
    
    error_function : Callable or None, default=None
        It is the function to be used when **error_method** is "custom". It
        takes two numpy arrays as input: the gt (first) and the predictions
        (second). Then, it computes the errors and return them as a numpy array.
    
    threshold_computation : ["gaussian", "mahalanobis", "custom"], default="gaussian"
        It is the way in which the threshold is computed over the errors
        computed between the gt and the prediction (remember, with gt we meant
        the value to be predicted or reconstructed, not the anomaly labels)
        of normal points. Moreover, all approaches estimating the standard
        deviation or covariance matrix use ddof=1 in the estimation and the
        errors vectors are interpreted to have samples on axis 0 and the
        features on axis 1. If the standard deviation is 0 or the sum of the
        elements in the covariance matrix is 0, the quantity 1e-10 is added to
        it. Here there is a description:
        
        - **gaussian**: it estimates the mean and covariance matrix (or standard
          deviation for scalar case). Then, it assumes a gaussian distribution
          for the errors and computes the inverse of the pdf for each vector
          (which should be errors on normal data). The threshold is selected as
          the greatest inverse pdf value between the errors.
        - **mahalanobis**: it estimates the mean and covariance matrix (or
          standard deviation for scalar case). Then, it computes the mahalanobis
          distance between each error vector and the mean. The threshold is
          selected as the greatest mahalanobis distance of the errors.
        - **custom**: custom function specified by the user.
    
    threshold_function : Callable or None, default=None
        It is the function to be used when **threshold_computation** is "custom".
        It takes one numpy array as input: the errors computed on normal points.
        Then, it computes the threshold and return it.
        
    scoring_function : ["gaussian", "mahalanobis"] or Callable, default="gaussian"
        It is the function used to score a single point. Its value depends on
        **threshold_computation** since the threshold is computed on the scores
        given to points. When **threshold_computation** is custom, it must be a
        callable function receiving a numpy array as input of shape (n_samples,
        n_features) and computes the scores which will be a numpy array of shape
        (n_samples). The input of this function are the errors computed by
        **error_function** if given or by using the method specified by
        **error_method**.

    Attributes
    ----------
    _mean : ndarray
        It is the vector of the mean computed on the errors of the validation
        set by calling `_learn_threshold()`.

    _cov : ndarray
        It is the covariance matrix computed on the errors of the validation set
        by calling `_learn_threshold()`. For one dimension, it is the standard
        deviation. When it is zero or the sum of all the elements of the matrix
        is zero, a small quantity is added to it to avoid zero division.

    _inv_cov : ndarray
        It is either the true inverse or the pseudo-inverse of the covariance
        matrix computed at the same time at which the covariance matrix is
        computed. For one dimension, it is the inverse of the standard deviation.

    _threshold : Number
        It is the threshold that has been found by means of learning on
        training and validation data by calling `_learn_threshold()`. The
        threshold is selected such that both training and validation are
        labelled as normal points.
    """
    _ACCEPTED_THRESHOLD = ["gaussian", "mahalanobis", "custom"]
    _ACCEPTED_ERROR = ["difference", "abs_difference", "norm", "custom"]
    __json_file = "tsa_ss_public.json"
    __numpy_file = "tsa_ss_private.npz"

    def __init__(self, error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian"):
        super().__init__()

        self.error_method = error_method
        self.error_function = error_function
        self.threshold_computation = threshold_computation
        self.threshold_function = threshold_function
        self.scoring_function = scoring_function

        self._mean = None
        self._cov = None
        self._inv_cov = None
        self._threshold = None

        self.__skip_checks_for_loading = False

        self.__check_parameters()

    def set_params(self, **params) -> None:
        super().set_params(**params)

        if not self.__skip_checks_for_loading:
            self.__check_parameters()

    def save(self, path: str,
             *args,
             **kwargs) -> None:
        """
        Parameters
        ----------
        path : str
            The path to the dir in which the model and all its related files
            must be saved.
        """
        find_or_create_dir(path)
        path_obj = Path(path)

        # create files to save the model parameters and attributes
        json_objects = self.get_params(deep=False)
        for key, value in json_objects.items():
            if isinstance(value, Callable):
                json_objects[key] = "warning"
        save_py_json(json_objects, str(path_obj / self.__json_file))

        if self._mean is not None:
            np.savez_compressed(str(path_obj / self.__numpy_file),
                                _mean=self._mean,
                                _cov=self._cov,
                                _inv_cov=self._inv_cov,
                                _threshold=np.array(self._threshold))

    def load(self, path: str,
             *args,
             **kwargs) -> None:
        """
        Parameters
        ----------
        path : str
            The path to the dir in which the model has been saved previously
            using method `save`.
        """
        self.__skip_checks_for_loading = True

        path_obj = Path(path)

        if not path_obj.joinpath(self.__json_file).is_file():
            raise ValueError("path directory is not valid. It must contain "
                             f"these files: {self.__json_file}")

        json_objects: dict = load_py_json(str(path_obj / self.__json_file))
        self.set_params(**json_objects)
        for key, value in self.get_params(deep=False).items():
            if value == "warning":
                self.set_params(**{key: None})
                print_warning(f"{key} was a callable. Set the callable to the "
                              "same callable used in the saved model.")

        if self.__numpy_file in os.listdir(str(path_obj)):
            with np.load(str(path_obj / self.__numpy_file)) as data:
                self._mean = data["_mean"]
                self._cov = data["_cov"]
                self._inv_cov = data["_inv_cov"]
                self._threshold = data["_threshold"]

        self.__skip_checks_for_loading = False

    def _compute_errors(self, gt, pred, verbose: bool = True, *args, **kwargs):
        """Compute the errors made from the prediction.
        
        Computes the errors between the ground truth and the prediction

        Parameters
        ----------
        gt : array-like of shape (n_samples, n_features)
            The ground truth.

        pred : array-like of shape (n_samples, n_features)
            The predictions of the ground truth.
            
        verbose : bool, default=True
            States if detailed printing must be performed.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        prediction_errors : ndarray of shape (n_samples, n_features)
            Errors of the prediction.
        """
        check_array(gt, force_all_finite="allow-nan")
        check_array(pred, force_all_finite="allow-nan")
        gt, pred = np.array(gt), np.array(pred)

        if gt.shape != pred.shape:
            raise ValueError("gt and pred must have the same shape")

        if verbose:
            print_step("Start to compute prediction errors")

        errors = np.zeros(gt.shape, dtype=np.double)
        # check values to ignore and use masked arrays to simplify operations
        gt = np.ma.array(gt, mask=np.isnan(gt), dtype=np.double)
        pred = np.ma.array(pred, mask=np.isnan(pred), dtype=np.double)

        if np.ma.count_masked(gt) != 0 or np.ma.count_masked(pred) != 0:
            print_warning("_compute_errors received gt and/or pred with nan "
                          "values. These positions will be ignored to compute "
                          "errors.")

        match self.error_method:
            case "difference":
                errors = gt - pred
                
            case "abs_difference":
                errors = np.abs(gt - pred)
            
            case "norm":
                # np.linalg.norm is not used such that masked arrays are always used
                errors = gt - pred
                errors = np.square(errors)
                errors = np.sum(errors, axis=1)
                errors = np.sqrt(errors).reshape((-1, 1))
                    
            case "custom":
                errors = self.error_function(gt, pred)
            
            case _:
                print_warning("The error method is wrongly set up and no "
                              "valid type of error has been chosen. The "
                              "errors will be all nan, but keep in mind that "
                              "this is just because errors_method is wrong.")
                errors = np.ma.masked_all(errors.shape)

        if verbose:
            print_step("Prediction errors have been computed")

        return errors.filled(np.nan)

    def _compute_scores(self, errors, verbose: bool = True, *args, **kwargs) -> np.ndarray:
        """Compute the anomaly scores for each error vector.
        
        Each error vector represent the error computed on a given point. The
        scores are computed from these errors.
        
        Parameters
        ----------
        errors : array-like of shape (n_samples, n_features)
            Data of the prediction errors.
            
        verbose : bool, default=True
            States if detailed printing must be performed.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The scores of the errors.
        """
        check_array(errors, force_all_finite="allow-nan")
        errors = np.ma.array(errors, mask=np.isnan(errors))
        rows_with_nan = np.unique(np.argwhere(np.ma.getmask(errors))[:, 0])
        rows_without_nan = np.array(sorted(set(range(errors.shape[0])).difference(rows_with_nan)))
        scores = np.full(errors.shape[0], fill_value=np.nan)
        clean_errors = errors[rows_without_nan]

        match self.threshold_computation:
            case "gaussian":
                # add 1e-10 to avoid zero division
                proba = multivariate_normal.pdf(clean_errors,
                                                mean=self._mean,
                                                cov=self._cov,
                                                allow_singular=True) + 1e-10
                valid_scores = 1 / proba
                scores[rows_without_nan] = valid_scores
                
            case "mahalanobis":
                valid_scores = np.array([mahalanobis(clean_errors[i], self._mean, self._inv_cov)
                                         for i in range(clean_errors.shape[0])])
                scores[rows_without_nan] = valid_scores
            
            case "custom":
                scores = self.scoring_function(errors)
            
            case _:
                print_warning("Impossible to compute score since the field "
                              "threshold_computation has an invalid value.")

        return scores

    def _compute_mean_and_cov(self, errors,
                              verbose: bool = True,
                              *args,
                              **kwargs) -> None:
        check_array(errors, force_all_finite="allow-nan")
        errors = np.ma.array(errors, mask=np.isnan(errors))

        self._mean = np.ma.mean(errors, axis=0)
        self._cov = np.ma.cov(errors, rowvar=False, ddof=1) if errors.ndim != 1 and errors.shape[1] != 1 else np.ma.std(errors, axis=0, ddof=1)
        is_vector = errors.ndim != 1 and errors.shape[1] != 1

        if (errors.ndim != 1 and errors.shape[1] != 1) and np.any(np.linalg.eigvals(self._cov) < 0):
            raise ValueError("Impossible to compute the covariance matrix.")

        if np.sum(self._cov) == 0:
            self._cov += 1e-10

        if is_vector:
            try:
                self._inv_cov = np.linalg.inv(self._cov)
            except LinAlgError:
                self._inv_cov = np.linalg.pinv(self._cov)
        else:
            self._inv_cov = 1 / self._cov

    def _learn_threshold(self, errors, verbose: bool = True, *args, **kwargs) -> None:
        """Computes the threshold to be used given the prediction errors.

        Remember to compute mean vector and covariance matrices before calling
        this method.

        Parameters
        ----------
        errors : array-like of shape (n_samples, n_features)
            Data of the prediction errors.
            
        verbose : bool, default=True
            States if detailed printing must be performed.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        check_array(errors, force_all_finite="allow-nan")
        errors = np.ma.array(errors, mask=np.isnan(errors))

        if verbose:
            print_step("Start to compute the threshold")
        
        match self.threshold_computation:
            case "gaussian" | "mahalanobis":
                scores = self._compute_scores(errors)
                scores = np.ma.array(scores, mask=np.isnan(scores))
                self._threshold = np.ma.max(scores)
                
            case "custom":
                self._threshold = self.threshold_function(errors)

            case _:
                print_warning("Impossible to compute the threshold since the "
                              "threshold_computation field has an invalid "
                              "value. Please, give it a valid value. The "
                              "threshold will be saved as nan.")
                self._threshold = np.nan

        if verbose:
            print_step("Threshold has been computed")

    def __check_parameters(self):
        check_argument_types([self.error_method,
                              self.error_function,
                              self.threshold_computation,
                              self.threshold_function,
                              self.scoring_function],
                             [str,
                              [Callable, None],
                              str,
                              [Callable, None],
                              [str, Callable]],
                             ["error_method",
                              "error_function",
                              "threshold_computation",
                              "threshold_function",
                              "scoring_function"])

        accepted_scoring = set(self._ACCEPTED_THRESHOLD).difference(["custom"])

        if self.error_method not in self._ACCEPTED_ERROR:
            raise ValueError(f"point_scoring must be one of {self._ACCEPTED_ERROR}")
        elif self.threshold_computation not in self._ACCEPTED_THRESHOLD:
            raise ValueError(f"threshold_computation must be one of {self._ACCEPTED_THRESHOLD}")
        elif self.scoring_function not in accepted_scoring and not isinstance(self.scoring_function, Callable):
            raise ValueError(f"scoring_function must be one of {self._ACCEPTED_THRESHOLD}")
        elif self.error_method == "custom" and self.error_function is None:
            raise ValueError("if error_method is custom, then, error_function "
                             "must be specified and must take two numpy arrays:"
                             " the gt (first) and the predictions (second). The"
                             " arrays are masked arrays.")
        elif self.threshold_computation == "custom" and self.threshold_function is None:
            raise ValueError("if threshold_computation is custom, then, "
                             "threshold_function must be specified and must take"
                             " one numpy array: the errors. The array is a "
                             "masked array.")
        elif self.threshold_computation == "custom" and not isinstance(self.scoring_function, Callable):
            raise ValueError("if threshold_computation is custom, then, the "
                             "scoring function must be specified as a callable")
