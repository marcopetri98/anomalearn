from sklearn.utils import check_array
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np


def analyse_stationarity(series,
                         method: str,
                         method_params: dict = None,
                         diff_order: int = 0) -> tuple[float, float, dict]:
    """Analyse the stationarity property of the time series.

    Parameters
    ----------
    series : array-like of shape (n_samples, n_features)
        The series to be analysed.

    method : ["adfuller", "kpss"]
        The stationarity test to be conducted on the time series.

    method_params : dict, default=None
        The additional parameters to the test method. The parameters are based
        on the test implementation which is available in `statsmodels` package.

    diff_order : int, default=0
        The number of times that the series must be differentiated. Normally, it
        is 0, which means that the series won't be differentiated. If it is `n`
        the series will be differentiated `n` times.

    Returns
    -------
    test_statistic : float
        The test statistic of the method for the time series.

    p_value : float
        The p-value of the method for the time series.
        
    critical_values : dict
        The critical values for the test statistic.

    Raises
    ------
    NotImplementedError
        If the method is not supported.
    """
    check_array(series, ensure_2d=False)
    series = np.asarray(series)
    
    if method_params is None:
        method_params = {}
    
    if diff_order > 0:
        series = np.diff(series, diff_order)
    
    match method:
        case "adfuller":
            test, p_value, _, _, critical_values, _ = adfuller(series, **method_params)
        
        case "kpss":
            test, p_value, _, critical_values = kpss(series, **method_params)
        
        case _:
            raise NotImplementedError(f"{method} not supported")
    
    return test, p_value, critical_values
