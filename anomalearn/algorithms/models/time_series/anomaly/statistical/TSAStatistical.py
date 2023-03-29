import abc
from typing import Callable, Tuple

import numpy as np
from sklearn.utils import check_array

from .. import TSAErrorBased
from ......utils.printing import print_step, print_header


class TSAStatistical(TSAErrorBased):
    """Abstract class for statistical models doing anomaly detection.

    This class includes all the classical statistical approaches based on
    prediction, such as SARMIAX or SES models. It is a wrapper of `statsmodels`
    or other statistical models adding the scoring, threshold and classification
    of points as anomalous or normal.

    These models are thought to predict a single point in the future. In fact,
    predicting multiple points in output will decrease the prediction accuracy
    and in some cases will also tend to the average as the horizon increases.
    However, when the horizon is greater than 1, the predictions are grouped in
    a vector `[p11, p12, ..., p1p, p21, ..., p2p, ..., pd1, ..., pdp]` where `d`
    is the number of dimensions and `p` is the prediction horizon. The ground
    truth to which the prediction vector is compared is a vector build as
    `[v1, v1, ..., v1, v2, ..., v2, ..., vd, ..., vd]`. Basically, the
    prediction vector at [i,j] has the prediction of feature `i` at time `t-j`
    and the ground truth vector has the feature values at time `t` repeated `l`
    times in order (first feature, second feature and so on).

    Parameters
    ----------
    prediction_horizon : int, default=1
        It is the number of future points to predict. Given points up to time
        `t`, if `prediction_horizon` is `n`, the points `t+1`, `t+2`, ..., `t+n`
        will be predicted by the model. Therefore, for each window it will
        predict the next `prediction_horizon` points.

    validation_split : float, default=0.1
        It is the percentage of training points to be used for validation. These
        points will be used to compute the errors on normal points and to
        compute the threshold to decide whether a point is an anomaly or not.

    mean_cov_sets : ["training", "validation", "both"], default="training"
        It states which set or sets must be used to compute the mean vector and
        covariance matrix to be used to compute the scores (if the scoring
        function requires them). With "training" the computation is performed
        on the training set, with "validation" the computation is performed on
        the validation set, with "both" the computation is performed on both of
        the sets.

    threshold_sets : ["training", "validation", "both"], default="training"
        It states which set or sets must be used to compute the threshold for
        classification. With "training" the computation is performed on the
        training set, with "validation" the computation is performed on  the
        validation set, with "both" the computation is performed on both of the
        sets.

    Attributes
    ----------
    _model
        It is the statistical model used to predict the next points.

    _fitted_model
        It is the statistical model fitted on training data to predict the next
        points. Its next points will be used to compute scores and for
        classification.
    """

    def __init__(self, prediction_horizon: int = 1,
                 validation_split: float = 0.1,
                 mean_cov_sets: str = "training",
                 threshold_sets: str = "training",
                 error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian"):
        super().__init__(error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function)

        self.prediction_horizon = prediction_horizon
        self.validation_split = validation_split
        self.mean_cov_sets = mean_cov_sets
        self.threshold_sets = threshold_sets

        self._model = None
        self._fitted_model = None

        self.__check_parameters()

    def set_params(self, **params) -> None:
        super().set_params(**params)
        self.__check_parameters()

    def fit(self, x=None,
            y=None,
            verbose: bool = True,
            fit_params: dict = None,
            build_params: dict = None,
            *args,
            **kwargs) -> None:
        """
        Parameters
        ----------
        verbose : bool, default=True
            If True, detailed printing of the process is performed. Otherwise,
            schematic printing is performed.

        fit_params : dict, default=None
            It is the dictionary of the fit arguments to pass to the wrapped
            model.

        build_params : dict, default=None
            It is the dictionary of the build arguments to pass to the wrapped
            model.
        """
        check_array(x)
        x = np.array(x)

        if verbose:
            print_header("Start of the model fit")

        if build_params is None:
            self._model_build()
        else:
            self._model_build(build_params=build_params)

        if verbose:
            print_step("Start to learn the parameters")

        if fit_params is None:
            self._model_fit()
        else:
            self._model_fit(fit_params=fit_params)

        if verbose:
            print_step("Parameters have been learnt")
            print_step("Predicting the time series")

        predictions, gt = self._get_predictions_and_gt(x)
        num_validation = round(x.shape[0] * self.validation_split)
        train_gt = gt[:num_validation]
        train_predictions = predictions[:num_validation]
        train_errors = self._compute_errors(train_gt, train_predictions, verbose=verbose)
        val_gt = gt[-num_validation:]
        val_predictions = predictions[-num_validation:]
        val_errors = self._compute_errors(val_gt, val_predictions, verbose=verbose)
        both_errors = self._compute_errors(gt, predictions, verbose=verbose)

        match self.mean_cov_sets:
            case "training":
                self._compute_mean_and_cov(train_errors, verbose=verbose)

            case "validation":
                self._compute_mean_and_cov(val_errors, verbose=verbose)

            case "both":
                self._compute_mean_and_cov(both_errors, verbose=verbose)

            case _:
                raise ValueError("mean_cov_sets has an invalid value")

        match self.threshold_sets:
            case "training":
                self._learn_threshold(train_errors, verbose=verbose)

            case "validation":
                self._learn_threshold(val_errors, verbose=verbose)

            case "both":
                self._learn_threshold(both_errors, verbose=verbose)

            case _:
                raise ValueError("threshold_sets has an invalid value")

        if verbose:
            print_step(f"The learnt threshold is {self._threshold}")
            print_header("End of the model fit")

    def predict(self, x,
                verbose: bool = True,
                *args,
                **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool, default=True
            If True, detailed printing of the process is performed. Otherwise,
            schematic printing is performed.
        """
        check_array(x)
        x = np.array(x)

        if verbose:
            print_header("Start of the predictions")
            print_step("Predicting points")

        predictions = self._model_predict(x)

        if verbose:
            print_header("Predictions ended")

        return predictions

    def classify(self, x,
                 verbose: bool = True,
                 *args,
                 **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool, default=True
            If True, detailed printing of the process is performed. Otherwise,
            synthetic printing is performed.
        """
        check_array(x)
        x = np.array(x)

        if verbose:
            print_header("Started points' classification")
            print_step("Predicting points and computing scores")

        predictions, gt = self._get_predictions_and_gt(x)
        errors = self._compute_errors(gt, predictions, verbose=verbose)
        scores = self._compute_scores(errors, verbose=verbose)

        if verbose:
            print_step("Classifying points")

        anomalies = np.argwhere(scores > self._threshold)
        pred_labels = np.zeros(x.shape[0], dtype=np.intc)
        pred_labels[anomalies] = 1

        if verbose:
            print_header("Points' classification ended")

        return pred_labels

    def anomaly_score(self, x,
                      verbose: bool = True,
                      *args,
                      **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool, default=True
            If True, detailed printing of the process is performed. Otherwise,
            synthetic printing is performed.
        """
        check_array(x)
        x = np.array(x)

        if verbose:
            print_header("Anomaly score computation started")
            print_step("Scoring points")

        predictions, gt = self._get_predictions_and_gt(x)
        errors = self._compute_errors(gt, predictions)
        scores = self._compute_scores(errors)

        if verbose:
            print_header("Anomaly score computation ended")

        return scores

    def _get_predictions_and_gt(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and gt.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            The time series to predict.

        Returns
        -------
        predictions, gt : ndarray, ndarray
            The predictions for each sample of the ground truth and the gt. When
            `prediction_horizon` is greater than 1, the predictions are in a 3D
            array that will be reshaped into a 2D array as in the class
            description. Same fot the gt.
        """
        predictions = self._model_predict(x)
        predictions = predictions.reshape((-1, predictions.shape[1] * predictions.shape[2]))
        gt = np.zeros(predictions.shape)
        for i in range(x.shape[1]):
            gt[:, i * self.prediction_horizon:i * self.prediction_horizon + self.prediction_horizon] = x[:, i].reshape((-1, 1))

        return predictions, gt

    @abc.abstractmethod
    def _model_predict(self, x: np.ndarray,
                       *args,
                       **kwargs) -> np.ndarray:
        """Predicts the values of x.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Data of the points to predict.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        predicted : ndarray of shape (n_samples, n_features, prediction_horizon)
            The predicted values for x. If the model requires a warm-up phase,
            some initial points may be `np.nan`.
        """

    def _model_fit(self, fit_params: dict = None,
                   *args,
                   **kwargs) -> None:
        """Fits the model to be trained.

        Parameters
        ----------
        fit_params : dict, default=None
            The parameters to pass to the fit function in case it is needed.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        if fit_params is None:
            self._fitted_model = self._model.fit()
        else:
            self._fitted_model = self._model.fit(**fit_params)

    @abc.abstractmethod
    def _model_build(self, build_params: dict = None,
                     *args,
                     **kwargs) -> None:
        """Builds the model.

        Parameters
        ----------
        build_params : dict, default=None
            The parameters to pass to the build function in case it is needed.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """

    def __check_parameters(self):
        """Checks that the class parameters are correct.

        Returns
        -------
        None
        """
        if not isinstance(self.prediction_horizon, int):
            raise TypeError("prediction_horizon must be a int")
        elif not isinstance(self.validation_split, float):
            raise TypeError("training_split must be a float")
        elif not isinstance(self.mean_cov_sets, str):
            raise TypeError("mean_cov_sets must be a string")
        elif not isinstance(self.threshold_sets, str):
            raise TypeError("threshold_sets must be string")

        mean_cov_sets = ["training", "validation", "both"]
        threshold_sets = ["training", "validation", "both"]

        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        elif not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be in range (0,1).")
        elif self.mean_cov_sets not in mean_cov_sets:
            raise ValueError(f"mean_cov_sets must be one of {mean_cov_sets}")
        elif self.threshold_sets not in threshold_sets:
            raise ValueError(f"threshold_sets must be one of {threshold_sets}")
