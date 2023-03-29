import abc
from typing import Tuple, Callable

import numpy as np
import tensorflow as tf

from . import TSANeuralNetwork
from ......utils import concat_list_array, print_warning


class TSANNStandard(TSANeuralNetwork):
    """Class representing anomaly detection based on prediction or reconstruction.
    
    Please, note that for statistical approaches to compute the threshold, the
    stride must always be 1. Otherwise, there will be points that won't be
    predicted and the probability density function cannot be computed.
    
    This class implements the creation of input vectors for neural network
    models predicting one or multiple points in the future or reconstructing
    subsequences of the time series. It also implements the functions used to
    compute the error vectors to score each point where the higher the score the
    more anomalous. Since the formation of the vector is identical in both cases,
    but the way in they are filled is different, the class defines an abstract
    method to be used for filling and implements the prediction and the method
    to actually predict the points.
    
    To compute the score the approach presented in Malhotra et al.
    (https://www.esann.org/proceedings/2015) and used also for reconstruction in
    (https://sites.google.com/site/icmlworkshoponanomalydetection/accepted-papers)
    from the same first author. On the subclasses it is specified how the vector
    is built specifically for prediction or reconstruction.
    """

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
        super().__init__(training_model=training_model,
                         prediction_model=prediction_model,
                         fitting_function=fitting_function,
                         prediction_horizon=prediction_horizon,
                         validation_split=validation_split,
                         mean_cov_sets=mean_cov_sets,
                         threshold_sets=threshold_sets,
                         error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function,
                         window=window,
                         stride=stride,
                         batch_size=batch_size,
                         stateful_model=stateful_model)

    def predict(self, x,
                verbose: bool = True,
                *args,
                **kwargs) -> np.ndarray:
        inputs, targets = self._build_x_y_sequences(x)
        pred, _ = self._get_pred_true_vectors(inputs, targets, x.shape[0])
        pred_orig = pred.reshape((pred.shape[0], targets.shape[1], -1))
        return np.nanmean(pred_orig, axis=2)

    def _remove_extra_points(self, samples: np.ndarray,
                             targets: np.ndarray,
                             verbose: bool = True,
                             *args,
                             **kwargs):
        """Remove extra points that does not fit into a batch.

        If the model must train in fixed batches of dimension batch_size, this
        function must be called. It removes all the extra points that does not
        fit into a batch, e.g., if `batch_size` is 11 and there are 100 samples,
        one sample will be discarded since with 9 batches we include 99 samples.
        The remaining sample cannot form a complete batch and will be discarded.

        If there aren't points to be discarded, this function does nothing.

        Parameters
        ----------
        samples : list of ndarray or ndarray of shape (n_samples, n_features, window)
            The samples built by `_build_x_y_sequences` method.

        targets : list of ndarray or ndarray of shape (n_samples, n_features, n_output)
            The targets built by `_build_x_y_sequences` method.

        verbose : bool, default=True
            States whether detailed printing should be performed.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        samples : ndarray of shape (n, n_features, window)
            The samples as a multiple of `batch_size`.

        targets : ndarray of shape (n, n_features, n_output)
            The targets as a multiple of `batch_size`.
        """
        # eliminate extra sequences in case must keep batches
        if samples.shape[0] % self.batch_size != 0:
            if verbose:
                print_warning("Some input samples will be discarded to keep the"
                              " number of points divisible by batch size")

            remainder = samples.shape[0] % self.batch_size
            samples = samples[:-remainder]
            targets = targets[:-remainder]

        return samples, targets
    
    def _get_pred_true_vectors(self, samples: list[np.ndarray] | np.ndarray,
                               targets: list[np.ndarray] | np.ndarray,
                               num_of_points: int | list[int],
                               verbose: bool = True,
                               *args,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        from_fit_multiple = isinstance(num_of_points, list)

        if from_fit_multiple:
            point_pred = [np.full((num_of_points[i], el.shape[1], el.shape[2]), np.nan) for i, el in enumerate(targets)]
            point_targ = [np.full((num_of_points[i], el.shape[1], el.shape[2]), np.nan) for i, el in enumerate(targets)]
        
            # build the predictions for
            for ser, (x, y) in enumerate(zip(samples, targets)):
                model_pred = self._prediction_model.predict(x)
                model_pred = model_pred.reshape(y.shape)
            
                self._fill_pred_targ_matrices(model_pred, y, point_pred[ser], point_targ[ser])
        
            all_pred = [el.reshape((el.shape[0], -1)) for el in point_pred]
            all_targ = [el.reshape((el.shape[0], -1)) for el in point_targ]
        
            point_pred = concat_list_array(all_pred)
            point_targ = concat_list_array(all_targ)
        else:
            point_pred = np.full((num_of_points, targets.shape[1], targets.shape[2]), np.nan)
            point_targ = point_pred.copy()
        
            model_pred = self._prediction_model.predict(samples)
            model_pred = model_pred.reshape(targets.shape)
        
            self._fill_pred_targ_matrices(model_pred, targets, point_pred, point_targ)

        point_pred = point_pred.reshape((point_pred.shape[0], -1))
        point_targ = point_targ.reshape((point_targ.shape[0], -1))
        return point_pred, point_targ
    
    @abc.abstractmethod
    def _fill_pred_targ_matrices(self, y_pred: np.ndarray,
                                 y_true: np.ndarray,
                                 mat_pred: np.ndarray,
                                 mat_true: np.ndarray) -> None:
        """Build target and prediction matrices.
        
        The function uses the fact that objects are passed by reference and
        automatically update the objects that are passed to it.
        
        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_features, output)
            It is the vector of the model's predictions.
        
        y_true : ndarray of shape (n_samples, n_features, output)
            It is the vector of the model's targets.
        
        mat_pred : ndarray of shape (n_points, n_features, output)
            It is the matrix containing the predictions for all the features
            at all horizons or of all reconstructing window for all time instants.
        
        mat_true : ndarray of shape (n_points, n_features, output)
            It is the matrix containing the true values for all the features
            at all horizons or of all reconstructing window for all time instants.

        Returns
        -------
        None
        """
        pass
