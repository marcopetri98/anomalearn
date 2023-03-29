from abc import ABC

import numpy as np
from sklearn.utils import check_array

from .ITimeSeriesAnomalyWrapper import ITimeSeriesAnomalyWrapper
from .TSAWindow import TSAWindow


class TSAWindowWrapper(TSAWindow, ITimeSeriesAnomalyWrapper, ABC):
    """Class representing an anomaly detector wrapping another method.

    Attributes
    ----------
    _wrapped_model
        The wrapped model that will be built and used to compute the scores for
        each window.
    """

    def __init__(self, window: int = 5,
                 stride: int = 1,
                 scaling: str = "minmax",
                 scoring: str = "average",
                 classification: str = "voting",
                 threshold: float = None,
                 anomaly_portion: float = 0.01):
        super().__init__(window=window,
                         stride=stride,
                         scaling=scaling,
                         scoring=scoring,
                         classification=classification,
                         threshold=threshold,
                         anomaly_portion=anomaly_portion)

        self._wrapped_model = None

    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        # Input validation
        check_array(x)
        x = np.array(x)

        # Projects the time series onto a vector space
        x_new, windows_per_point = self._project_time_series(x)

        # Get the window scores
        window_scores = self._compute_window_scores(x_new)
        anomaly_scores = self._compute_point_scores(window_scores,
                                                    windows_per_point)
        return anomaly_scores

    def classify(self, x, *args, **kwargs) -> np.ndarray:
        # Input validation
        check_array(x)
        X = np.array(x)

        # Projects the time series onto a vector space
        x_new, windows_per_point = self._project_time_series(X)

        # Get window labels
        window_scores = self._compute_window_scores(x_new)
        window_anomalies = self._compute_window_labels(x_new)
        anomaly_scores = self._compute_point_scores(window_scores,
                                                    windows_per_point)
        labels, _ = self._compute_point_labels(window_anomalies,
                                               windows_per_point,
                                               anomaly_scores)
        return labels
