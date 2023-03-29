from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

from anomalearn.algorithms.models.time_series.anomaly.deep_learning import TSANNStandard


class FakeModel(object):
    def __init__(self, output_shape: Tuple):
        super().__init__()

        self.output_shape = output_shape
        self.number_of_times_predict_called = 0

    def reset_num_predict_called(self):
        self.number_of_times_predict_called = 0

    def predict(self, x):
        self.number_of_times_predict_called += 1
        return np.random.rand(*self.output_shape)


class TSANNStandardChild(TSANNStandard):
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
                 max_epochs: int = 100,
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

        self.build_x_y_sequences_called = 0
        self.get_pred_true_vectors_called = 0

    def reset_counters(self):
        self.build_x_y_sequences_called = 0
        self.get_pred_true_vectors_called = 0

    def _get_pred_true_vectors(self, samples: list[np.ndarray] | np.ndarray,
                               targets: list[np.ndarray] | np.ndarray,
                               num_of_points: int | list[int],
                               verbose: bool = True,
                               *args,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        self.get_pred_true_vectors_called += 1

        return super()._get_pred_true_vectors(samples, targets, num_of_points, verbose, *args, **kwargs)

    def _fill_pred_targ_matrices(self, y_pred: np.ndarray,
                                 y_true: np.ndarray,
                                 mat_pred: np.ndarray,
                                 mat_true: np.ndarray) -> None:
        mat_pred = np.random.rand(*mat_pred.shape)
        mat_true = np.random.rand(*mat_true.shape)

    def _build_x_y_sequences(self, x: np.ndarray,
                             keep_batches: bool = False,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        self.build_x_y_sequences_called += 1

        samples, targets = None, None

        for i in range(0, x.shape[0] - self.window + 1, self.stride):
            new_sample = np.transpose(x[i:i+self.window]).reshape((1, x.shape[1], self.window))

            if samples is None:
                samples = new_sample
                targets = new_sample
            else:
                samples = np.concatenate((samples, new_sample))
                targets = np.concatenate((targets, new_sample))

        return samples, targets
