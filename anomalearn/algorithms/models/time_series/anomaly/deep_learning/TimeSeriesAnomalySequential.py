from abc import ABC
from typing import Tuple

import numpy as np
from sklearn.utils import check_array

from .TimeSeriesAnomalyWindowDL import TimeSeriesAnomalyWindowDL


class TimeSeriesAnomalySequential(TimeSeriesAnomalyWindowDL, ABC):
    """Represent DL models performing sequential analysis of the input."""
    
    def __init__(self, window: int = 200,
                 stride: int = 1,
                 forecast: int = 1,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 predict_validation: float = 0.2,
                 batch_divide_training: bool = False,
                 folder_save_path: str = "data/nn_models/",
                 filename: str = "sequential",
                 distribution: str = "gaussian",
                 perc_quantile: float = 0.999):
        super().__init__(window=window,
                         stride=stride,
                         forecast=forecast,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         predict_validation=predict_validation,
                         batch_divide_training=batch_divide_training,
                         folder_save_path=folder_save_path,
                         filename=filename,
                         distribution=distribution,
                         perc_quantile=perc_quantile)
    
    def _build_x_y_sequences(self, x) -> Tuple[np.ndarray, np.ndarray]:
        samples = []
        targets = []
        
        for i in range(0, x.shape[0] - self.window - self.forecast, self.stride):
            samples.append(x[i:i + self.window])
            targets.append(x[i + self.window:i + self.window + self.forecast])
        
        return np.array(samples), np.array(targets)
    
    def predict_time_series(self, xp, x) -> np.ndarray:
        check_array(xp)
        check_array(x)
        xp = np.array(xp)
        x = np.array(x)
        
        if xp.shape[0] < self.window:
            raise ValueError("You must provide at lest window points to predict")
        
        return self._predict_future(x[-self.window:], x)
