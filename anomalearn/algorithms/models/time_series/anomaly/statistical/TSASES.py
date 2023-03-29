from typing import Callable

import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from .TSAES import TSAES


class TSASES(TSAES):
    """SES model to perform anomaly detection on time series.
    
    The SES implemented by statsmodels is the Holt-Winters definition of Simple
    Exponential Smoothing, which is a specific case of Exponential smoothing.
    For more details check the `statsmodels <https://www.statsmodels.org/stable/
    api.html>` implementation.
    
    When points are predicted using SES, it is important to know that the points
    to be predicted must be the whole sequence immediately after the training.
    It is not possible to predicted from the 50th point to the 100th with the
    current implementation.
    
    This model is a subclass of :class:`~models.time_series.anomaly.statistical
    .TimeSeriesAnomalyES` since SES is conceptually the same thing of ES where
    we have neither trend nor seasonality.
    
    Notes
    -----
    For all the other parameters that are not included in the documentation, see
    the `statsmodels <https://www.statsmodels.org/stable/api.html>`
    documentation for `SES models <https://www.statsmodels.org/stable/generated/
    statsmodels.tsa.holtwinters.SimpleExpSmoothing.html>`."""

    def __init__(self, prediction_horizon: int = 1,
                 validation_split: float = 0.1,
                 mean_cov_sets: str = "training",
                 threshold_sets: str = "training",
                 error_method: str = "difference",
                 error_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 threshold_computation: str = "gaussian",
                 threshold_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 scoring_function: str | Callable[[np.ndarray], np.ndarray] = "gaussian"):
        super().__init__(prediction_horizon=prediction_horizon,
                         validation_split=validation_split,
                         mean_cov_sets=mean_cov_sets,
                         threshold_sets=threshold_sets,
                         error_method=error_method,
                         error_function=error_function,
                         threshold_computation=threshold_computation,
                         threshold_function=threshold_function,
                         scoring_function=scoring_function)

    def _model_build(self, build_params: dict = None,
                     *args,
                     **kwargs) -> None:
        if build_params is None:
            raise ValueError("simple exponential smoothing needs endog in order to be created")

        self._model = SimpleExpSmoothing(**build_params)
