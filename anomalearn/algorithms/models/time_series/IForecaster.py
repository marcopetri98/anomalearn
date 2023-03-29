import abc
from abc import ABC

import numpy as np


class IForecaster(ABC):
    """Interface identifying a forecaster of a time-series.
    """
    
    @abc.abstractmethod
    def forecast(self, x,
                 steps: int = 1,
                 *args,
                 **kwargs) -> np.ndarray:
        """Forecast values for the time-series.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The time series from which values must be forecasted.
            
        steps : int, default=1
            The number of forecasting steps to perform.
        
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        forecasted_values : ndarray of shape (steps, n_features)
            The forecasted values for the time series points.
        """
        pass
