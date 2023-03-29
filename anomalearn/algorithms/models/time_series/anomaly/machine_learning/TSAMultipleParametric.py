from abc import ABC

import numpy as np
from sklearn.utils import check_array

from . import TSAWindowWrapper
from .....IMultipleParametric import IMultipleParametric


class TSAMultipleParametric(TSAWindowWrapper, IMultipleParametric, ABC):
    """A machine learning AD multiple parametric model."""

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

    def fit(self, x, y=None, *args, **kwargs) -> None:
        check_array(x)
        x = np.array(x)

        x_new, windows_per_point = self._project_time_series(x)
        self._build_wrapped()
        self._wrapped_model.fit(x_new)

    def fit_multiple(self, x: list, y: list = None, *args, **kwargs) -> None:
        if len(x) == 0:
            raise ValueError("x must have at least one element!")
        else:
            for l in x:
                check_array(l)

        x_total = None
        x_new = list()
        windows_per_point = list()
        for series in x:
            ser_new, ser_windows_per_point = self._project_time_series(series)
            x_new.append(ser_new)
            windows_per_point.append(ser_windows_per_point)

            if x_total is None:
                x_total = ser_new
            else:
                x_total = np.append(x_total, ser_new, axis=0)

        self._build_wrapped()
        self._wrapped_model.fit(x_total)
