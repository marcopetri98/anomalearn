from __future__ import annotations

from pathlib import Path
from typing import Tuple

from sklearn.utils import check_array
from skopt.space import Integer
import numpy as np

from ...utils import load_py_json, save_py_json
from .. import IShapeChanger
from ..pipelines import AbstractPipelineSavableLayer


class SlidingWindowForecast(IShapeChanger, AbstractPipelineSavableLayer):
    """Preprocessing object transforming the input using the sliding window technique.
    
    The object takes as input a time series, thus an array-like with two
    dimensions.
    
    Parameters
    ----------
    window : int
        It is the window to be used while doing the sliding window.
    
    stride : int, default=1
        It is the stride to be used while doing the sliding window.
        
    forecast : int, default=1
        It is the number of forecast to consider in building the sliding windows.
        
    Attributes
    ----------
    _points_seen : int
        It is the number of points seen during the last `shape_change` call.
    """
    __json_file = "sliding_window_forecast.json"
    
    def __init__(self, window: int = 10,
                 stride: int = 1,
                 forecast: int = 1):
        super().__init__()
        
        self.window = window
        self.stride = stride
        self.forecast = forecast
        
        self._points_seen = None
        
    @property
    def points_seen(self):
        """Gets the number of points seen in the last call to `shape_change`.
        
        Returns
        -------
        points_seen
            The number of points seen in the last call to `shape_change`.
        """
        return self._points_seen

    def get_hyperparameters(self, *args, **kwargs) -> dict:
        return {"window": {"value": self.window, "set": Integer(1, np.inf)},
                "stride": {"value": self.stride, "set": Integer(1, np.inf)},
                "forecast": {"value": self.forecast, "set": Integer(1, np.inf)}}

    def set_hyperparameters(self, hyperparameters: dict, *args, **kwargs) -> None:
        self.set_params(**hyperparameters)
        
    def __repr__(self):
        return f"SlidingWindowForecast(window={self.window}, stride={self.stride}, forecast={self.forecast})"
    
    def __str__(self):
        return "SlidingWindowForecast"
    
    def __eq__(self, other):
        if not isinstance(other, SlidingWindowForecast):
            return False
        
        return (self.window, self.stride, self.forecast) == (other.window, other.stride, other.forecast)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def copy(self) -> SlidingWindowForecast:
        """Copies the object.
        
        Returns
        -------
        new_obj : SlidingWindowForecast
            A new object identical to this.
        """
        new = SlidingWindowForecast(window=self.window,
                                    stride=self.stride,
                                    forecast=self.forecast)
        new._points_seen = self._points_seen
        return new
    
    def save(self, path,
             *args,
             **kwargs) -> SlidingWindowForecast:
        super().save(path=path)
        path_obj = Path(path)
        
        save_py_json({"_points_seen": self._points_seen}, str(path_obj / self.__json_file))
        return self
    
    def load(self, path: str,
             *args,
             **kwargs) -> SlidingWindowForecast:
        super().load(path=path)
        path_obj = Path(path)
        
        self._points_seen = load_py_json(str(path_obj / self.__json_file))["_points_seen"]
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> SlidingWindowForecast:
        obj = SlidingWindowForecast()
        obj.load(path)
        return obj
    
    def get_input_shape(self) -> tuple:
        return "n", "m"
    
    def get_output_shape(self) -> tuple:
        return "n", self.forecast, "m"
        
    def shape_change(self, x, y=None, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Builds input and target vectors doing sliding window.
        
        The sliding window is performed such that a window has as targets the
        following `forecast` points.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            It is the data to be reshaped using the sliding window approach.
            
        y
            Ignored.
            
        Returns
        -------
        x_new : array-like
            The sliding windows of the time series with given window and stride.
            
        y_new : array-like
            The windows containing the next `forecast` points after `x_new`.
        """
        check_array(x)
        x = np.array(x)
    
        self._points_seen = x.shape[0]
        window_data = None
        window_target = None
        for i in range(0, x.shape[0] - self.window - self.forecast + 1, self.stride):
            new_data = x[i:i + self.window].reshape((1, self.window, x.shape[1]))
            new_target = x[i + self.window:i + self.window + self.forecast].reshape((1, self.forecast, x.shape[1]))
            
            if window_data is None:
                window_data = new_data.copy()
                window_target = new_target.copy()
            else:
                window_data = np.concatenate((window_data, new_data))
                window_target = np.concatenate((window_target, new_target))
                
        return window_data, window_target
