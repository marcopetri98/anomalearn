from typing import Callable

import numpy as np

from anomalearn.algorithms.models.time_series.anomaly import TSAErrorBased


class TSAErrorBasedChild(TSAErrorBased):
    def __init__(self, error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian"):
        super().__init__(error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function)

    def fit(self, x, y=None, *args, **kwargs) -> None:
        pass
