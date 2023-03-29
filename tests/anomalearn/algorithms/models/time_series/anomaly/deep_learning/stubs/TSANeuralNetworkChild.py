from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

from anomalearn.algorithms.models.time_series.anomaly.deep_learning import TSANeuralNetwork


class TSANeuralNetworkChild(TSANeuralNetwork):
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

        self.fake_scores = False
        self.num_of_points_called = 0
        self.total_samples_input = 0
        self.skip_fit_and_complete = False
        self.fit_and_complete_args = None
        self.fit_and_complete_kwargs = None

    def reset_flags_counters(self):
        self.fake_scores = False
        self.num_of_points_called = 0
        self.total_samples_input = 0
        self.skip_fit_and_complete = False
        self.fit_and_complete_args = None
        self.fit_and_complete_kwargs = None

    def anomaly_score(self, x,
                      verbose: bool = True,
                      *args,
                      **kwargs) -> np.ndarray:
        if self.fake_scores:
            return np.arange(x.shape[0])
        else:
            return super().anomaly_score(x, verbose)

    def _fit_and_complete(self, samples_train: np.ndarray,
                          targets_train: np.ndarray,
                          samples_valid: np.ndarray,
                          targets_valid: np.ndarray,
                          callbacks: list[tf.keras.callbacks.Callback],
                          num_of_points: int | list[int],
                          verbose: bool = True,
                          *args,
                          **kwargs) -> None:
        if not self.skip_fit_and_complete:
            super()._fit_and_complete(samples_train=samples_train,
                                      targets_train=targets_train,
                                      samples_valid=samples_valid,
                                      targets_valid=targets_valid,
                                      callbacks=callbacks,
                                      num_of_points=num_of_points,
                                      verbose=verbose,
                                      *args,
                                      **kwargs)
        else:
            self.fit_and_complete_args = [samples_train, targets_train, samples_valid, targets_valid, callbacks, num_of_points, verbose]
            self.fit_and_complete_kwargs = kwargs

    def predict(self, x,
                *args,
                **kwargs) -> np.ndarray:
        return np.random.rand(*x.shape)

    def _get_pred_true_vectors(self, samples: list[np.ndarray] | np.ndarray,
                               targets: list[np.ndarray] | np.ndarray,
                               num_of_points: int | list[int],
                               verbose: bool = True,
                               *args,
                               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        n = num_of_points if isinstance(num_of_points, int) else sum(num_of_points)
        self.num_of_points_called += n

        try:
            pred, true = np.random.rand(n, targets.shape[1] * targets.shape[2]), np.random.rand(n, targets.shape[1] * targets.shape[2])
            self.total_samples_input += samples.shape[0]
        except AttributeError:
            pred, true = np.random.rand(n, targets[0].shape[1] * targets[0].shape[2]), np.random.rand(n, targets[0].shape[1] * targets[0].shape[2])
            self.total_samples_input += sum([el.shape[0] for el in samples])

        return pred, true

    def _build_x_y_sequences(self, x: np.ndarray,
                             keep_batches: bool = False,
                             verbose: bool = True,
                             *args,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if self.stateful_model:
            num = (x.shape[0] // self.batch_size) * self.batch_size
        else:
            num = x.shape[0]

        return np.random.rand(num, x.shape[1], self.window), np.random.rand(num, x.shape[1], self.window)
