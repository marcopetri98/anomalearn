from __future__ import annotations

import abc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Callable

import numpy as np
import tensorflow as tf
from sklearn.utils import check_array

from .. import TSAErrorBased
from .... import IAnomalyScorer, IAnomalyClassifier
from ..... import IMultipleParametric
from ......utils import print_warning, print_header, print_step, save_py_json, \
    load_py_json, concat_list_array


class StatesResetAtSpecifiedBatches(tf.keras.callbacks.Callback):
    """Resets the states of the model before certain batches are evaluated.
    
    The callback has been introduced to deal with learning on multiple sequences
    not related with each other. When a specified batch is observed, the method
    `reset_states` of the model is called.
    """
    
    def __init__(self, train_batches: list,
                 valid_batches: list):
        super().__init__()
        
        self.train_batches = train_batches
        self.valid_batches = valid_batches
        
    def on_train_batch_begin(self, batch, logs=None):
        if batch in self.train_batches:
            self.model.reset_states()
    
    def on_test_batch_begin(self, batch, logs=None):
        if batch in self.valid_batches:
            self.model.reset_states()


class TSANeuralNetwork(IAnomalyScorer, IAnomalyClassifier, TSAErrorBased, IMultipleParametric):
    """Abstract class grouping common anomaly detection deep pieces.
    
    It is important to keep in mind that this class does not set a seed! If you
    want reproducible results, set `numpy` and `tensorflow` seeds manually.
    
    This class represent any approach that could be used to perform anomaly
    detection on time series. Currently, it considers methods common to its
    subclasses, and it is designed to be compatible and used with `tensorflow`.
    
    Parameters
    ----------
    training_model : tf.keras.Model
        It is the model to be used for training. Note that the model will be
        cloned to avoid exposing members of the object.
    
    prediction_model : tf.keras.Model
        It is the model to be used for prediction. This model is identical to
        `training_model` in most of the cases, the difference may be that of the
        batch size in input. Stateful models (containing stateful layers such as
        RNN or LSTM) require the specification of the batch size in input, which
        we might be different between prediction and training. Note that the
        model will be cloned to avoid exposing members of the object.
    
    fitting_function : Callable[[Model, ndarray, ndarray, ndarray, ndarray, int, int, bool, list], tf.keras.callbacks.History]
        The fitting function is the function that will be called to fit the
        model. It receives the following arguments (in this order):
        
        - model: the keras model that the function must fit.
        - x_train: the training inputs.
        - y_train: the training targets.
        - x_val: the validation inputs.
        - y_val: the validation targets.
        - batch_size: the batch_size.
        - callbacks: a list of the needed callbacks (other callbacks can be
          added to it).
          
        Then, after the fitting has been performed, the fitting_function must
        return the history of the training for the model. Since the models are
        cloned when they are passed to this class, the fitting function must
        also compile the model to fit it.
    
    prediction_horizon : int, default=1
        It is the number of future points to predict. Given points up to time
        `t`, if `prediction_horizon` is `n`, the points `t+1`, `t+2`, ..., `t+n`
        will be predicted by the model. Therefore, for each window it will
        predict the next `prediction_horizon` points. This field may be useless
        for methods that are not predictive such as the auto-encoders.

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
    _initial_training_model : Model
        It is the `training_model` passed when the object has been created. This
        member serves just as model that will be used for training when multiple
        fit functions are called by cloning it to `_training_model` which will
        actually be trained. Moreover, it is the training model that will be
        serialized to file such that when the class is loaded, the fit function
        can be called immediately.
    
    _training_model : Model
        It is the tensorflow keras model used for actual training.
    
    _prediction_model : Model
        It is the `prediction_model` passed when the object has been created. It
        won't be used for training, it will receive weights from
        `_training_model` when the fitting will be completed. This distinction
        has been made to allow the usage of different batch sizes between
        training and prediction.
    
    _fit_history : dict
        It is the history of the model fit. Keep in mind that if the method `fit`
        is called multiple times, the history dictionary will be updated
        multiple times, thus, it will keep only the history of the latest
        training.
    """
    __history_file = "tsa_nn_history.json"
    __training_model_dir = "tsa_nn_training_model"
    __trained_model_dir = "tsa_nn_trained_model"
    
    def __init__(self, training_model: tf.keras.Model,
                 prediction_model: tf.keras.Model,
                 fitting_function: Callable[[tf.keras.Model,
                                             np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             int,
                                             list], tf.keras.callbacks.History],
                 prediction_horizon: int = 1,
                 validation_split: float = 0.1,
                 mean_cov_sets: str = "training",
                 threshold_sets: str = "training",
                 error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian",
                 *,
                 window: int = 200,
                 stride: int = 1,
                 batch_size: int = 32,
                 stateful_model: bool = False):
        super().__init__(error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function)

        self.fitting_function: Callable = fitting_function
        self.prediction_horizon = prediction_horizon
        self.validation_split = validation_split
        self.mean_cov_sets = mean_cov_sets
        self.threshold_sets = threshold_sets
        
        self.window = window
        self.stride = stride
        self.batch_size = batch_size
        self.stateful_model = stateful_model

        self._initial_training_model: tf.keras.Model = tf.keras.models.clone_model(training_model)
        self._training_model: tf.keras.Model = tf.keras.models.clone_model(training_model)
        self._prediction_model: tf.keras.Model = tf.keras.models.clone_model(prediction_model)
        self._fit_history: dict | None = None

        self.__skip_checks_for_loading = False
        
        self.__check_parameters()
        
    def set_params(self, **params) -> None:
        super().set_params(**params)

        if not self.__skip_checks_for_loading:
            self.__check_parameters()
        
    def save(self, path: str,
             *args,
             **kwargs) -> None:
        super().save(path=path)
        
        path_obj = Path(path)
        
        save_py_json(self._fit_history, str(path_obj / self.__history_file))
        self._initial_training_model.save(str(path_obj / self.__training_model_dir))
        self._prediction_model.save(str(path_obj / self.__trained_model_dir))
    
    def load(self, path: str,
             *args,
             **kwargs) -> None:
        self.__skip_checks_for_loading = True

        super().load(path=path)
        
        path_obj = Path(path)
        
        self._fit_history = load_py_json(str(path_obj / self.__history_file))
        self._initial_training_model = tf.keras.models.load_model(str(path_obj / self.__training_model_dir))
        self._training_model = tf.keras.models.clone_model(self._initial_training_model)
        self._prediction_model = tf.keras.models.load_model(str(path_obj / self.__trained_model_dir))

        self.__skip_checks_for_loading = False
        
    def anomaly_score(self, x,
                      verbose: bool = True,
                      *args,
                      **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool, default=True
            States whether detailed printing should be performed.
        """
        inputs, targets = self._build_x_y_sequences(x, verbose=verbose)
        pred, true = self._get_pred_true_vectors(inputs, targets, x.shape[0], verbose=verbose)
        errors = self._compute_errors(true, pred, verbose=verbose)
        return self._compute_scores(errors, verbose=verbose)

    def classify(self, x,
                 verbose: bool = True,
                 *args,
                 **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        verbose : bool, default=True
            States whether detailed printing should be performed.
        """
        scores = self.anomaly_score(x, verbose=verbose)
        return scores > self._threshold
        
    def fit_multiple(self, x: list,
                     y: list = None,
                     verbose: bool = True,
                     *args,
                     **kwargs) -> None:
        """
        The validation set is retrieved at the end of the list. This is because
        the method assumes that the training samples come from the same time
        series, thus, they are ordered with respect to it. The early stopping
        set is retrieved at the end of each sequence of training.
        
        Parameters
        ----------
        verbose : bool, default=True
            States whether detailed printing should be performed.
        """
        if not isinstance(x, list):
            raise ValueError("x must be a list of array-like")
        
        if len(x) == 0:
            raise ValueError("x must have at least one element")
        else:
            for el in x:
                check_array(el)
                
        if verbose:
            print_header("Start TSANeuralNetwork fit")
            print_step("Start to build training sequences")
            
        # build a list of all the training sequences
        samples = None
        targets = None
        num_pts = 0
        for series in x:
            series = np.array(series)
            xx, yy = self._build_x_y_sequences(series,
                                               keep_batches=True,
                                               verbose=verbose)
            num_pts += xx.shape[0]
            
            if samples is None or targets is None:
                samples = [xx]
                targets = [yy]
            else:
                samples.append(xx)
                targets.append(yy)
                
        if verbose:
            print_step("Start to build training and validation sets")
            
        # compute the number of points in validation and training sequences
        # to be used
        samples_train = []
        targets_train = []
        samples_valid = []
        targets_valid = []
        
        needed_valid = round(self.validation_split * num_pts)
        
        if needed_valid < self.batch_size:
            raise ValueError("the validation is so small that it cannot even "
                             "contain a single batch")
            
        # split values
        for sample, target in zip(reversed(samples), reversed(targets)):
            if sample.shape[0] <= needed_valid:
                # inserts a full series as validation
                samples_valid.insert(0, sample)
                targets_valid.insert(0, target)
                needed_valid -= sample.shape[0]
            elif needed_valid != 0:
                # split a series with a part of training and a part of validation
                valid_batches = needed_valid // self.batch_size
                valid_points = valid_batches * self.batch_size
                
                if valid_batches != needed_valid / self.batch_size and verbose:
                    print_warning("In the construction of the validation set, "
                                  "some points must be discarded to keep the "
                                  "batch size.")

                samples_valid.insert(0, sample[-valid_points:])
                targets_valid.insert(0, target[-valid_points:])
                samples_train.insert(0, sample[:-valid_points])
                targets_train.insert(0, target[:-valid_points])
                needed_valid = 0
            else:
                # fill the training
                samples_train.insert(0, sample)
                targets_train.insert(0, target)
            
        if verbose:
            print_step("Continue with actual model training")
        
        # build the list of batches at which states must be reset
        train_reset = []
        valid_reset = []
        for train, valid in zip(samples_train, samples_valid):
            if len(train_reset) == 0:
                train_reset.append(train.shape[0] // self.batch_size)
                valid_reset.append(valid.shape[0] // self.batch_size)
            else:
                train_reset.append(train_reset[-1] + train.shape[0] // self.batch_size)
                valid_reset.append(valid_reset[-1] + valid.shape[0] // self.batch_size)
        
        # compute final training and validation arrays
        train_samples, train_targets = concat_list_array(samples_train), concat_list_array(targets_train)
        valid_samples, valid_targets = concat_list_array(samples_valid), concat_list_array(targets_valid)
        callbacks = [StatesResetAtSpecifiedBatches(train_reset, valid_reset)]

        self._fit_and_complete(samples_train=train_samples,
                               targets_train=train_targets,
                               samples_valid=valid_samples,
                               targets_valid=valid_targets,
                               callbacks=callbacks,
                               num_of_points=[el.shape[0] for el in x],
                               verbose=verbose,
                               samples_train_list=samples_train,
                               targets_train_list=targets_train,
                               samples_valid_list=samples_valid,
                               targets_valid_list=targets_valid)
        
        if verbose:
            print_header("End of TSANeuralNetwork fit")
        
    def fit(self, x,
            y=None,
            verbose: bool = True,
            *args,
            **kwargs) -> None:
        """
        Parameters
        ----------
        verbose : bool, default=True
            States whether detailed printing should be performed.
        """
        if verbose:
            print_header("Start TSANeuralNetwork fit")
            print_step("Building training sequence")
            
        # divide train points for model and train points for threshold
        samples, targets = self._build_x_y_sequences(x,
                                                     keep_batches=self.stateful_model,
                                                     verbose=verbose)

        valid_pts = round(self.validation_split * samples.shape[0])
        if self.stateful_model:
            valid_batches = valid_pts // self.batch_size
    
            if valid_batches == 0:
                raise ValueError("the validation set is so small that it "
                                 "cannot even contain a single batch")
            elif valid_batches != valid_pts / self.batch_size and verbose:
                print_warning("In the construction of the validation "
                              "set, some points will be discarded")
    
            valid_pts = valid_batches * self.batch_size

        if verbose:
            print_step(f"The validation contains {valid_pts} points")

        samples_train = samples[:-valid_pts]
        targets_train = targets[:-valid_pts]
        samples_valid = samples[-valid_pts:]
        targets_valid = targets[-valid_pts:]

        self._fit_and_complete(samples_train=samples_train,
                               targets_train=targets_train,
                               samples_valid=samples_valid,
                               targets_valid=targets_valid,
                               callbacks=[],
                               num_of_points=x.shape[0],
                               verbose=verbose)

        if verbose:
            print_header("End of TSANeuralNetwork fit")
            
    def _fit_and_complete(self, samples_train: np.ndarray,
                          targets_train: np.ndarray,
                          samples_valid: np.ndarray,
                          targets_valid: np.ndarray,
                          callbacks: list[tf.keras.callbacks.Callback],
                          num_of_points: int | list[int],
                          verbose: bool = True,
                          *args,
                          **kwargs) -> None:
        """Fits the model and call super methods for threshold estimation.

        This function receives the pre-processed points with their targets for
        training and validation. Then, it calls the fitting function over the
        training model, set up the prediction model and calls the methods to
        learn the threshold and covariance matrix and mean vector.
        
        Parameters
        ----------
        samples_train : ndarray
            Points for the training.
        
        targets_train : ndarray
            Targets for the training points.
        
        samples_valid : ndarray
            Points for the validation.
        
        targets_valid : ndarray
            Targets for the validation points.
        
        callbacks : list[Callback]
            The callbacks for the training.

        num_of_points : int | list[int]
            The number of points of the original time series to analyse or a
            list of the number of points in case of multiple fit.
        
        verbose : bool
            States if detailed printing must be performed.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Contains list with training and validation series in case it has
            been called from `fit_multiple`.

        Returns
        -------
        None
        """
        self._training_model = tf.keras.models.clone_model(self._initial_training_model)
        history = self.fitting_function(self._training_model,
                                        samples_train,
                                        targets_train,
                                        samples_valid,
                                        targets_valid,
                                        self.batch_size,
                                        callbacks)
        self._fit_history = history.history

        with TemporaryDirectory() as temp_dir:
            self._training_model.save_weights(temp_dir)
            self._prediction_model.load_weights(temp_dir).expect_partial()

        if "samples_train_list" in kwargs.keys():
            samples_train = kwargs["samples_train_list"]
            targets_train = kwargs["targets_train_list"]
            samples_valid = kwargs["samples_valid_list"]
            targets_valid = kwargs["targets_valid_list"]
            samples_all = samples_train + samples_valid
            targets_all = targets_train + targets_valid
        else:
            samples_all = np.concatenate((samples_train, samples_valid))
            targets_all = np.concatenate((targets_train, targets_valid))
    
        match self.mean_cov_sets:
            case "training":
                self._call_compute_mean_cov(samples=samples_train,
                                            targets=targets_train,
                                            num_of_points=num_of_points,
                                            verbose=verbose)
                
            case "validation":
                self._call_compute_mean_cov(samples=samples_valid,
                                            targets=targets_valid,
                                            num_of_points=num_of_points,
                                            verbose=verbose)
            
            case "both":
                self._call_compute_mean_cov(samples=samples_all,
                                            targets=targets_all,
                                            num_of_points=num_of_points,
                                            verbose=verbose)
    
        match self.threshold_sets:
            case "training":
                self._call_learn_threshold(samples=samples_train,
                                           targets=targets_train,
                                           num_of_points=num_of_points,
                                           verbose=verbose)
                
            case "validation":
                self._call_learn_threshold(samples=samples_valid,
                                           targets=targets_valid,
                                           num_of_points=num_of_points,
                                           verbose=verbose)
            
            case "both":
                self._call_learn_threshold(samples=samples_all,
                                           targets=targets_all,
                                           num_of_points=num_of_points,
                                           verbose=verbose)
    
    def _call_compute_mean_cov(self, samples: list[np.ndarray] | np.ndarray,
                               targets: list[np.ndarray] | np.ndarray,
                               num_of_points: int | list[int],
                               verbose: bool = True,
                               *args,
                               **kwargs) -> None:
        """Builds the vectors and calls the method to learn matrix and vector.
        
        Parameters
        ----------
        samples : list of ndarray or ndarray of shape (n_samples, n_features, window)
            The set to be used to learn how to score and classify each point.
        
        targets : list of ndarray or ndarray of shape (n_samples, n_features, n_output)
            The targets for the set used to allow the computation of the errors
            with respect to the prediction.

        num_of_points : int | list[int]
            The number of points of the original time series to analyse or a
            list of the number of points in case of multiple fit.
        
        verbose : bool, default=True
            States whether detailed printing should be performed.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        point_pred, point_targ = self._get_pred_true_vectors(samples,
                                                             targets,
                                                             num_of_points,
                                                             verbose)
        errors = self._compute_errors(point_targ, point_pred, verbose=verbose)
        self._compute_mean_and_cov(errors, verbose=verbose)
    
    def _call_learn_threshold(self, samples: list[np.ndarray] | np.ndarray,
                              targets: list[np.ndarray] | np.ndarray,
                              num_of_points: int | list[int],
                              verbose: bool = True,
                              *args,
                              **kwargs) -> None:
        """Builds the vectors and calls the method to learn the threshold.
        
        Parameters
        ----------
        samples : list of ndarray or ndarray of shape (n_samples, n_features, window)
            The set to be used to learn how to score and classify each point.
        
        targets : list of ndarray or ndarray of shape (n_samples, n_features, n_output)
            The targets for the set used to allow the computation of the errors
            with respect to the prediction.

        num_of_points : int | list[int]
            The number of points of the original time series to analyse or a
            list of the number of points in case of multiple fit.
        
        verbose : bool, default=True
            States whether detailed printing should be performed.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        point_pred, point_targ = self._get_pred_true_vectors(samples,
                                                             targets,
                                                             num_of_points,
                                                             verbose)
        errors = self._compute_errors(point_targ, point_pred, verbose=verbose)
        self._learn_threshold(errors, verbose=verbose)

    @abc.abstractmethod
    def _get_pred_true_vectors(self, samples: list[np.ndarray] | np.ndarray,
                               targets: list[np.ndarray] | np.ndarray,
                               num_of_points: int | list[int],
                               verbose: bool = True,
                               *args,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the vectors of the predictions and related ground truth.
        
        Parameters
        ----------
        samples : list of ndarray or ndarray of shape (n_samples, n_features, window)
            The set to be used to learn how to score and classify each point.
        
        targets : list of ndarray or ndarray of shape (n_samples, n_features, n_output)
            The targets for the set used to allow the computation of the errors
            with respect to the prediction.

        num_of_points : int | list[int]
            The number of points of the original time series to analyse or a
            list of the number of points in case of multiple fit.
        
        verbose : bool, default=True
            States whether detailed printing should be performed.
        
        args
            Not used, present to allow multiple inheritance and signature change.
        
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        pred : ndarray of shape (n, n_components)
            It is the vector of the predictions for each time instant. For
            prediction model n_components should be `n_features * horizon` while
            for auto-encoders it should be `n_features * n_window`.
        
        true : ndarray of shape (n, n_components)
            It is the target vector for the predictions for each time instant.
            For prediction model n_components should be `n_features * horizon`
            while for auto-encoders it should be `n_features * n_window`.
        """
        pass

    @abc.abstractmethod
    def _build_x_y_sequences(self, x: np.ndarray,
                             keep_batches: bool = False,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Build the neural network inputs to perform regression.
        
        The function computes the samples and the targets starting from x. When
        `keep_batches` is `True` the function returns two lists which can be
        divided by `batch_size`.

        Parameters
        ----------
        x : ndarray
            The training input sequence.
            
        keep_batches : bool, default=False
            States if the created sequences must be such that the final number
            of points can be divided by batch_size.

        Returns
        -------
        x_train : ndarray of shape (n_samples, n_features, window_size)
            An array with the training samples for the model.
            
        y_train : ndarray of shape (n_samples, n_features, n_output)
            An array with the target samples for the model.
            
        Raises
        ------
        ValueError
            If from the input sequence it is impossible to extract training
            samples due to the window and stride configuration.
            
        Warnings
        --------
        If the model is stateful and the sequence is not a multiple of the batch
        a warning is printed to let the user know that some points have been
        discarded.
        """
        pass
        
    def __check_parameters(self):
        if not isinstance(self.fitting_function, Callable):
            raise TypeError("fitting_function must be a callable")
        elif not isinstance(self.prediction_horizon, int):
            raise TypeError("prediction_horizon must be int")
        elif not isinstance(self.validation_split, float):
            raise TypeError("validation_split must be float")
        elif not isinstance(self.mean_cov_sets, str):
            raise TypeError("mean_cov_sets must be str")
        elif not isinstance(self.threshold_sets, str):
            raise TypeError("threshold_sets must be str")
        elif not isinstance(self.window, int):
            raise TypeError("window must be int")
        elif not isinstance(self.stride, int):
            raise TypeError("stride must be int")
        elif not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be int")
        elif not isinstance(self.stateful_model, bool):
            raise TypeError("stateful_model must be bool")

        mean_cov_sets = ["training", "validation", "both"]
        threshold_sets = ["training", "validation", "both"]
        
        # single variable errors
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be greater than 1")
        elif not 0 < self.validation_split < 1:
            raise ValueError("early_stopping_split must lie inside (0,1)")
        elif self.window < 1:
            raise ValueError("window must be greater than 0")
        elif self.stride < 1:
            raise ValueError("stride must be greater than 0")
        elif self.batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        elif self.mean_cov_sets not in mean_cov_sets:
            raise ValueError(f"mean_cov_sets must be one of {mean_cov_sets}")
        elif self.threshold_sets not in threshold_sets:
            raise ValueError(f"threshold_sets must be one of {threshold_sets}")

        # values errors
        if self.stride != 1 and self.stateful_model:
            print_warning("Stride is not 1 and the model is stateful. It may "
                          "cause problems for learning.")
