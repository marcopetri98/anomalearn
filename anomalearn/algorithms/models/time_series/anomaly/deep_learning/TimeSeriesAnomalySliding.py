from abc import ABC

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .TimeSeriesAnomalySequential import TimeSeriesAnomalySequential


class TimeSeriesAnomalySliding(TimeSeriesAnomalySequential, ABC):
    """DL model to perform anomaly detection with sequential sliding window."""
    
    def __init__(self, window: int = 200,
                 stride: int = 1,
                 forecast: int = 1,
                 batch_size: int = 32,
                 max_epochs: int = 50,
                 predict_validation: float = 0.2,
                 batch_divide_training: bool = False,
                 folder_save_path: str = "data/nn_models/",
                 filename: str = "sliding",
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
    
    def _predict_future(self, xp: np.ndarray, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["model_"])
        check_array(xp)
        xp = np.array(xp.copy())
        
        # I build the array of inputs for the model
        inputs = [xp.copy()]
        for i in range(0, x.shape[0], self.forecast):
            # Concatenate to the previous window the next points
            xp = np.concatenate((xp[self.forecast:], x[i:i + self.forecast]),
                                axis=0)
            inputs.append(xp.copy())
        inputs = np.array(inputs)
        
        predictions = self.model_.predict(inputs, batch_size=1)
        predictions = predictions.reshape((predictions.shape[0], 1))
        
        if predictions.shape[0] > x.shape[0]:
            to_discard = predictions.shape[0] - x.shape[0]
            predictions = predictions[:-to_discard]
        
        return predictions
