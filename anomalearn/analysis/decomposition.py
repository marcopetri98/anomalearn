from sklearn.utils import check_array
from statsmodels.tsa.seasonal import STL, DecomposeResult, seasonal_decompose
import numpy as np


def decompose_time_series(series,
                          method: str,
                          method_params: dict = None,
                          diff_order: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose the time series into trend-cycly, seasonal and residual.

    Parameters
    ----------
    series : array-like of shape (n_samples, n_features)
        The series to be analysed.

    method : ["stl", "moving_average"]
        The decomposition method used to decompose the time series.

    method_params : dict, default=None
        The additional parameters to the decomposition method. The parameters
        are based on the decomposition implementation which is available in
        `statsmodels` package.

    diff_order : int, default=0
        The number of times that the series must be differentiated. Normally, it
        is 0, which means that the series won't be differentiated. If it is `n`
        the series will be differentiated `n` times.

    Returns
    -------
    trend_cycle : ndarray
        The trend-cycle component of the time series.

    seasonal : ndarray
        The seasonal component of the time series.

    residual : ndarray
        The residual component of the time series.
    """
    check_array(series, ensure_2d=False)
    series = np.asarray(series)
    
    if method_params is None:
        method_params = {}
    
    if diff_order > 0:
        series = np.diff(series, diff_order)
    
    match method:
        case "stl":
            res: DecomposeResult = STL(series, **method_params).fit()
            seasonal, trend_cycle, residual = res.seasonal, res.trend, res.resid
        
        case "moving_average":
            res: DecomposeResult = seasonal_decompose(series, **method_params)
            seasonal, trend_cycle, residual = res.seasonal, res.trend, res.resid
        
        case _:
            raise NotImplementedError(f"{method} not supported")
    
    return trend_cycle, seasonal, residual
