from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

from . import TSANNStandard


class TSANNReconstructor(TSANNStandard):
    """Class representing anomaly detection based on reconstruction.
    
    Please, note that for statistical approaches to compute the threshold, the
    stride must always be 1. Otherwise, there will be points that won't be
    predicted and the probability density function cannot be computed.
    
    This class implements the creation of input vectors for neural network
    models implementing a reconstruction approach. Moreover, it also implements
    the functions used to compute the error vectors to score each point where
    the higher the score the more anomalous.
    
    The error vectors are computed as in Malhotra et al. (https://sites.google.com/site/icmlworkshoponanomalydetection/accepted-papers)
    considering the overlapping between points to build the error. Namely, with
    a window of `w` and `d` features, the error vectors for the reconstructions
    are `[e11, e12, ..., e1w, e21, ..., e2w, ..., ed1, ..., edw]` from
    predictions `[p11, p12, ..., p1w, p21, ..., pdw]` where the element `eij` is
    the error vector of feature `i` reconstructed from the window starting at
    `t-1-j`. The point `pkq` is the point of feature `k` reconstructed from the
    window starting at `t-1-q`. The ground truth to compare with the predictions
    is `[v1, v1, ..., v1 (w times), ...]` where `vi` is the value of feature `i`
    at time `t`.
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

    def _build_x_y_sequences(self, x: np.ndarray,
                             keep_batches: bool = False,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        samples = None
        targets = None

        # build all the possible sequences
        for i in range(0, x.shape[0] - self.window + 1, self.stride):
            new_sample = np.transpose(np.array(x[i:i + self.window])).reshape((1, x.shape[1], self.window))
            new_target = np.transpose(np.array(x[i:i + self.window])).reshape((1, x.shape[1], self.window))
            
            if samples is None or targets is None:
                samples = new_sample
                targets = new_target
            else:
                samples = np.concatenate((samples, new_sample))
                targets = np.concatenate((targets, new_target))

        if keep_batches:
            samples, targets = self._remove_extra_points(samples=samples, targets=targets, verbose=verbose)
                
        return samples, targets

    def _fill_pred_targ_matrices(self, y_pred: np.ndarray,
                                 y_true: np.ndarray,
                                 mat_pred: np.ndarray,
                                 mat_true: np.ndarray) -> None:
        """Build target and reconstruction matrices.

        The function uses the fact that objects are passed by reference and
        automatically update the objects that are passed to it.

        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_features, horizon)
            It is the vector of the model's predictions.

        y_true : ndarray of shape (n_samples, n_features, horizon)
            It is the vector of the model's targets.

        mat_pred : ndarray of shape (n_points, n_features, horizon)
            It is the matrix containing the predictions for all the features
            from all reconstructing windows.

        mat_true : ndarray of shape (n_points, n_features, horizon)
            It is the matrix containing the true values for all the features
            for all reconstructed windows.

        Returns
        -------
        None
        """
        for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
            # get index of first and last predicted points for slicing
            start_idx = idx * self.stride
            end_idx = start_idx + self.window
        
            # fill the matrices with the predictions
            for f in range(pred.shape[0]):
                np.fill_diagonal(mat_pred[start_idx:end_idx, f, :], pred[f, :])
                np.fill_diagonal(mat_true[start_idx:end_idx, f, :], true[f, :])
