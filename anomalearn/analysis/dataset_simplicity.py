import logging
import math
import warnings
from typing import Callable, Any, Optional

import numpy as np
from numba import jit, prange
from sklearn.utils import check_array, check_X_y

from ..exceptions import ClosedOpenRangeError
from ..utils import mov_avg, mov_std
from ..utils.metrics import _true_positive_rate, _true_negative_rate


__module_logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True)
def _find_best_constants(channel: np.ndarray,
                         desc: np.ndarray,
                         asc: np.ndarray,
                         labels: np.ndarray) -> tuple[float, float]:
    """Finds the best constants to detect anomalies on channel with tnr=1.
    
    Parameters
    ----------
    channel : np.ndarray of shape (n_samples)
        The channel on which the constants must be found.
    
    desc : np.ndarray of shape (n_samples)
        The values of the channel from in descending order.
    
    asc : np.ndarray of shape (n_samples)
        The values of the channel from in ascending order.
    
    labels : np.ndarray of shape (n_samples)
        The labels of the channel.

    Returns
    -------
    lower : float
        The lower threshold.
    
    up : float
        The upper threshold.
    """
    up, low = math.nan, math.nan
    
    # find best bounds
    tnr_up, tpr_up, score_up = 1, 0, 0
    tnr_dw, tpr_dw, score_dw = 1, 0, 0
    for i in range(desc.shape[0]):
        if tnr_up == 1:
            curr_pred = (channel >= desc[i]).reshape(-1)
            tpr_up = _true_positive_rate(labels, curr_pred)
            tnr_up = _true_negative_rate(labels, curr_pred)
            if tpr_up > score_up and tnr_up == 1:
                up = desc[i]
                score_up = tpr_up
            
        if tnr_dw == 1:
            curr_pred = (channel <= asc[i]).reshape(-1)
            tpr_dw = _true_positive_rate(labels, curr_pred)
            tnr_dw = _true_negative_rate(labels, curr_pred)
            if tpr_dw > score_dw and tnr_dw == 1:
                low = asc[i]
                score_dw = tpr_dw
        
        if tnr_up != 1 and tnr_dw != 1:
            break
        
    return low, up


@jit(nopython=True, parallel=True)
def _transpose_numpy(x: np.ndarray) -> np.ndarray:
    """Transposes `x` and copies it such that C order is maintained.
    
    Parameters
    ----------
    x : ndarray of shape (n, m)
        The numpy array to be transposed.

    Returns
    -------
    transposed_x : ndarray of shape (m, n)
        The transposed copy of `x`.
    """
    transposed_x = np.zeros((x.shape[1], x.shape[0]), dtype=x.dtype)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            transposed_x[j, i] = x[i, j]
    return transposed_x


@jit(nopython=True)
def _diff_numpy(x: np.ndarray,
                n: int) -> np.ndarray:
    """Computes the array of differences.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_feature)
        The array to be differentiated.
    
    n : int
        The order of the differentiation.

    Returns
    -------
    diff_array : ndarray
        The differentiated array.
    """
    transposed_x = _transpose_numpy(x)
    series = np.diff(transposed_x, n)
    series = _transpose_numpy(series)
    return series


@jit(nopython=True, parallel=True)
def _find_constant_score(x: np.ndarray,
                         y: np.ndarray) -> tuple[float, list[float], list[float]]:
    """Find the constant score on this dataset.
    
    This method is exhaustive and find the exact constant score. If the constant
    score is 1 the dataset is constant simple.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        The time series to analyse.
    
    y : ndarray of shape (n_samples,)
        The labels of the time series.

    Returns
    -------
    constant_score : float
        The constant score of the time series.
        
    upper_bound : list[float]
        The upper bounds at which values equal or higher are considered
        anomalous.
        
    lower_bound : list[float]
        The lower bounds at which values equal or lower are considered
        anomalous.
    """
    # transpose manually the array since numba does not accept axis on sort and
    # flip. Moreover, it does not compile by using x.transpose() since the
    # elements will be Fortran continuous and not C continuous.
    transposed_x = _transpose_numpy(x)
    asc_x = np.sort(transposed_x)
    desc_x = np.fliplr(asc_x)
    asc_x = _transpose_numpy(asc_x)
    desc_x = _transpose_numpy(desc_x)

    # find the best constants and score feature-wise
    c_up = [math.nan] * x.shape[1]
    c_low = [math.nan] * x.shape[1]
    for f in range(x.shape[1]):
        c_low[f], c_up[f] = _find_best_constants(x[:, f], desc_x[:, f], asc_x[:, f], y)
    
    # find the best score overall
    pred = np.full(y.shape, False, dtype=np.bool_)
    for f in range(x.shape[1]):
        if c_up[f] is not None and c_low[f] is not None:
            pred = pred | ((x[:, f] >= c_up[f]) | (x[:, f] <= c_low[f])).reshape(-1)
        elif c_up[f] is not None and c_low[f] is None:
            pred = pred | (x[:, f] >= c_up[f]).reshape(-1)
        elif c_up[f] is None and c_low[f] is not None:
            pred = pred | (x[:, f] <= c_low[f]).reshape(-1)
    pred = pred.reshape(-1)
    
    constant_score = _true_positive_rate(y, pred)
    return constant_score, c_up, c_low


def _get_windows_to_try(window_range: tuple[int, int] | slice | list[int] = (2, 300)):
    """Converts window ranges into a list of windows to try.
    
    Parameters
    ----------
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try.

    Returns
    -------
    windows_to_try : list[int]
        The list of all windows to try given the window range.
    """
    if not isinstance(window_range, tuple) and not isinstance(window_range, slice) and not isinstance(window_range, list):
        raise TypeError("window_range must be a tuple, slice or list")
    
    if isinstance(window_range, tuple):
        if len(window_range) != 2:
            raise ValueError("window_range must be a tuple of two elements")
        elif window_range[0] > window_range[1]:
            raise ValueError("window_range[0] must be less or equal than "
                             f"window_range[1]. Received {window_range}")
        elif window_range[0] < 2:
            raise ValueError("window_range[0] must be at least 2")
    elif isinstance(window_range, list) and len(window_range) == 0:
        raise ValueError("window_range must have elements if it is a list")
    
    if isinstance(window_range, list):
        return window_range
    elif isinstance(window_range, slice):
        return [w for w in range(window_range.start, window_range.stop, window_range.step)]
    else:
        windows = []
        i = window_range[0]
        while i <= window_range[1]:
            windows.append(i)
        
            if i < 5:
                i += 1
            elif 5 <= i < 20:
                i += 5
            elif 20 <= i < 100:
                i += 10
            elif 100 <= i < 200:
                i += 20
            elif 200 <= i < 300:
                i += 50
            else:
                i += 10 ** math.floor(math.log10(i))
        
        if window_range[1] not in windows:
            windows.append(window_range[1])
        return windows


# TODO: continue to check until numba.typed.List and numba.typed.Dict will become stable and change
@jit(nopython=True)
def _analyse_constant_simplicity(x: np.ndarray,
                                 y: np.ndarray,
                                 diff: int = 3) -> tuple[float, list[float], list[float], int]:
    """Same as exposed analyse constant simplicity but numba compatible.
    
    Parameters
    ----------
    x : np.ndarray of shape (n_samples, n_features)
        The same as analyse constant simplicity.
        
    y : np.ndarray of shape (n_samples)
        The same as analyse constant simplicity.
        
    diff : int, default=3
        The same as analyse constant simplicity.

    Returns
    -------
    best_constant_score : float
        The best constant score of the time series.
        
    best_upper_bound : list[float]
        The best upper bounds at which values equal or higher are considered
        anomalous.
        
    best_lower_bound : list[float]
        The best lower bounds at which values equal or lower are considered
        anomalous.
        
    best_diff_order : int
        The best time series differencing order to achieve the constant score.
    """
    best_score = math.nan
    best_upper = [math.nan] * x.shape[1]
    best_lower = [math.nan] * x.shape[1]
    best_diff = -1
    diffs = diff + 1
    for diff_order in range(diffs):
        # stop if differencing eliminated all the anomalies
        if len(np.unique(y[diff_order:])) != 2:
            break
        
        series = x
        if diff_order != 0:
            series = _diff_numpy(x, diff_order)
        
        score, upper, lower = _find_constant_score(series, y[diff_order:])
        if math.isnan(best_score):
            best_score = score
            best_upper = upper
            best_lower = lower
            best_diff = diff_order
        elif score > best_score:
            best_score = score
            best_upper = upper
            best_lower = lower
            best_diff = diff_order
        
        if best_score == 1:
            break
    
    return best_score, best_upper, best_lower, best_diff


def _fix_numba_upper_lower(best_upper: list,
                           best_lower: list) -> tuple[list, list]:
    """Converts float nan to None.
    
    Parameters
    ----------
    best_upper : list
        The upper bounds computed by compiled functions.
    
    best_lower : list
        The lower bounds computed by compiled functions.

    Returns
    -------
    corrected_upper_bounds : list
        The upper bound passed to the functions in which float nans are
        converted to None.
    
    corrected_lower_bounds : list
        The lower bound passed to the functions in which float nans are
        converted to None.
    """
    final_upper, final_lower = [], []
    for up, low in zip(best_upper, best_lower):
        if math.isnan(up):
            final_upper.append(None)
        else:
            final_upper.append(up)
        
        if math.isnan(low):
            final_lower.append(None)
        else:
            final_lower.append(low)
    return final_upper, final_lower


def _check_analysis_inputs(x,
                           y,
                           diff: int,
                           window_range: tuple[int, int] | slice | list[int] | None = None) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    """Checks that inputs are ok and process them.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed passed to analysis functions.

    y : array-like of shape (n_samples,)
        The labels of the time series passed to analysis functions.

    diff : int
        The diff parameter passed to the analysis functions.
    
    window_range : tuple[int, int] or slice or list[int], default=None
        The window range passed to the analysis functions. If None, it means
        that the windows to try must not be computed and None will be returned.

    Returns
    -------
    new_x : ndarray
        The numpy ndarray of the input x.
    
    new_y : ndarray
        The numpy ndarray of the input y.
    
    windows_to_try : list[int]
        The list of all windows to try. None if `window_range` is None.
    """
    if not isinstance(diff, int):
        raise TypeError("diff must be an integer")
    
    if diff < 0:
        raise ClosedOpenRangeError(0, np.inf, diff)
    
    if window_range is not None:
        windows_to_try = _get_windows_to_try(window_range)
    else:
        windows_to_try = None
    
    check_array(x, force_all_finite="allow-nan")
    check_X_y(x, y, force_all_finite="allow-nan")
    x = np.array(x)
    y = np.array(y)
    
    __module_logger.debug(f"np.unique(y)={np.unique(y)}")
    if len(np.unique(y)) != 2:
        raise ValueError("The input labels are more than 2, there must be only "
                         "two labels: 0 (negative) and 1 (positive).")
    elif 0 not in y or 1 not in y:
        raise ValueError("The input labels must be 0 (negative) and 1 (positive)")
    
    return x, y, windows_to_try


def analyse_constant_simplicity(x, y, diff: int = 3) -> dict:
    """Analyses whether the series is constant simple and its score.

    A dataset is constant simple if just by using one constant for each
    channel it is possible to separate normal and anomalous points. This
    function analyses this property and gives a score of 1 when the dataset
    is constant simple. It gives 0 when no anomalies can be found without
    producing false positives. Therefore, the higher the score, the higher
    the number of anomalies that can be found just by placing a constant.
    The score is the True Positive Rate (TPR) at True Negative Rate (TNR)
    equal to 1.

    The analysis tries to divide the normal and anomalous points just by
    placing a constant. Basically, it states that the anomalous and normal
    points can be perfectly divided one from the other.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.

    y : array-like of shape (n_samples,)
        The labels of the time series.

    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has four keys:
        `constant_score`, `upper_bound`, `lower_bound` and `diff_order`. The
        former is the constant score which lies between 0 and 1. The second is
        the upper bound for each channel that have been found to separate
        anomalies and normal points. The third is the lower bound for each
        channel. The latter is the differencing applied to the time series to
        find the constant score and constants. If any of the bounds is None it
        means that there is no separation keeping TNR at 1. When bounds are not
        None, a point is labelled anomalous if any feature of that point is
        greater or lower than found bounds.
    """
    x, y, _ = _check_analysis_inputs(x, y, diff)
    __module_logger.warning("currently typed.List and typed.Dict for numba are "
                            "still not used")
    best_score, best_upper, best_lower, best_diff = _analyse_constant_simplicity(x, y, diff)
    final_upper, final_lower = _fix_numba_upper_lower(best_upper, best_lower)
    
    return {"constant_score": best_score, "upper_bound": final_upper, "lower_bound": final_lower, "diff_order": best_diff}


# TODO: continue to check until numba.typed.List and numba.typed.Dict will become stable and change
@jit(nopython=True)
def _fast_execute_movement_simplicity(x,
                                      y,
                                      diff: int,
                                      windows_to_try: list[int],
                                      statistical_movement: Callable[[Any, int, Optional[str]], np.ndarray]) -> tuple[float, list[float], list[float], int, int]:
    """Computes the movement simplicity score.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Same as `_execute_movement_simplicity`.

    y : array-like of shape (n_samples,)
        Same as `_execute_movement_simplicity`.

    diff : int, default=3
        Same as `_execute_movement_simplicity`.

    windows_to_try : list[int]
        The windows to try to compute the movement simplicity score.

    statistical_movement : Callable[[Any, int, Optional[str]], np.ndarray]
        Same as `_execute_movement_simplicity`.

    Returns
    -------
    best_constant_score : float
        The best constant score of the time series.
        
    best_upper_bound : list[float]
        The best upper bounds at which values equal or higher are considered
        anomalous.
        
    best_lower_bound : list[float]
        The best lower bounds at which values equal or lower are considered
        anomalous.
        
    best_diff_order : int
        The best time series differencing order to achieve the constant score.
        
    best_window : int
        The best window for classifying the time series using the given
        statistical movement.
    """
    best_score = math.nan
    best_upper = [math.nan] * x.shape[1]
    best_lower = [math.nan] * x.shape[1]
    best_diff = -1
    best_window = -1
    diffs = diff + 1
    for diff_order in range(diffs):
        # stop if differencing eliminated all the anomalies
        if len(np.unique(y[diff_order:])) != 2:
            break
        
        series = x
        labels = y
        if diff_order != 0:
            series = _diff_numpy(x, diff_order)
            labels = y[diff_order:]
        
        for window in windows_to_try:
            movement = statistical_movement(series, window, "right")
            score, upper, lower, _ = _analyse_constant_simplicity(movement, labels, 0)
            
            if math.isnan(best_score):
                best_score = score
                best_upper = upper
                best_lower = lower
                best_diff = diff_order
                best_window = window
            elif score > best_score:
                best_score = score
                best_upper = upper
                best_lower = lower
                best_diff = diff_order
                best_window = window
            
            if best_score == 1:
                break
    
    return best_score, best_upper, best_lower, best_diff, best_window


def _execute_movement_simplicity(x,
                                 y,
                                 diff: int,
                                 window_range: tuple[int, int] | slice | list[int],
                                 statistical_movement: Callable[[Any, int, Optional[str]], np.ndarray]) -> dict:
    """Computes the movement simplicity score.

    A movement is considered a statistical quantity computed over a sliding
    window on the original time series.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.

    y : array-like of shape (n_samples,)
        The labels of the time series.

    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.

    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.

    statistical_movement : Callable[[Any, int, Optional[str]], np.ndarray]
        It is the function producing the time series of the statistical movement
        to be checked for simplicity.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has four keys:
        `movement_score`, `upper_bound`, `lower_bound`, `window` and `diff_order`.
        The former is the movement score which lies between 0 and 1. The second
        is the upper bound for each channel that have been found to separate
        anomalies and normal points. The third is the lower bound for each
        channel. The fourth is the window. The latter is the differencing
        applied to the time series to find the constant score and constants. If
        any of the bounds is None it means that there is no separation keeping
        TNR at 1. When bounds are not None, a point is labelled anomalous if
        any feature of that point is greater or lower than found bounds.
    """
    if not isinstance(statistical_movement, Callable):
        raise TypeError("statistical_movement must be Callable[[Any, int, str], np.ndarray]")
    
    x, y, windows_to_try = _check_analysis_inputs(x, y, diff, window_range)
    __module_logger.warning("currently typed.List and typed.Dict for numba are "
                            "still not used")
    best_score, best_upper, best_lower, best_diff, best_window = _fast_execute_movement_simplicity(x, y, diff, windows_to_try, statistical_movement)
    final_upper, final_lower = _fix_numba_upper_lower(best_upper, best_lower)
    
    if best_diff == -1:
        warnings.warn("no configuration can be tested on this input", RuntimeWarning)
    
    return {"movement_score": best_score, "upper_bound": final_upper, "lower_bound": final_lower, "diff_order": best_diff, "window": best_window}


def analyse_mov_avg_simplicity(x,
                               y,
                               diff: int = 3,
                               window_range: tuple[int, int] | slice | list[int] = (2, 300)) -> dict:
    """Analyses whether the time series is moving average simple and its score.
    
    A dataset is moving average simple if just by placing a constant over
    the moving average series it is possible to separate normal and
    anomalous points. Here, the considered moving average can have window
    lengths all different for all channels. The function gives a sore of 1
    when the dataset is moving average simple. It gives 0 when no anomalies
    can be found without producing false positives. Therefore, the higher
    the score, the higher the number of anomalies that can be found just
    by placing a constant on the moving average series. The score is the
    True Positive Rate (TPR) at True Negative Rate (TNR) equal to 1.
    
    The analysis tries to divide the normal and anomalous points just by
    placing a constant in a moving average space of the time series. It
    means that the time series is first projected into a moving average
    space with window of length `w` to be found, then a constant to divide
    the points is found. If the time series is multivariate, the constant
    will be a constant vector in which elements are the constants for the
    time series channels and each channel may be projected with a different
    window `w`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.
        
    y : array-like of shape (n_samples,)
        The labels of the time series.
        
    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.
        
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has three keys:
        `mov_avg_score`, `upper_bound`, `lower_bound`, `window` and `diff_order`.
        The first is the moving average score which lies between 0 and 1. The
        second is the upper bound for each channel that have been found to
        separate anomalies and normal points. The third is the lower bound for
        each channel. The fourth is the best window that has been found to yield
        the score. The last is the differencing that has been applied to the
        time series before searching for the moving average score. If any of the
        bounds is None it means that there is no separation keeping TNR at 1.
        When bounds are not None, a point is labelled anomalous if any
        feature of the moving average series of that point is greater or
        lower than found bounds.
    """
    result = _execute_movement_simplicity(x, y, diff, window_range, mov_avg)
    result["mov_avg_score"] = result["movement_score"]
    del result["movement_score"]
    return result


def analyse_mov_std_simplicity(x,
                               y,
                               diff: int = 3,
                               window_range: tuple[int, int] | slice | list[int] = (2, 300)) -> dict:
    """Analyses whether the time series is moving standard deviation simple and its score.
    
    A dataset is moving standard deviation simple if just by placing a
    constant over  the moving standard deviation series it is possible to
    separate normal and anomalous points. Here, the considered moving
    standard deviation can have window lengths all different for all
    channels. The function gives a sore of 1  when the dataset is moving
    standard deviation simple. It gives 0 when no anomalies can be found
    without producing false positives. Therefore, the higher the score, the
    higher the number of anomalies that can be found just by placing a
    constant on the moving average series. The score is the True Positive
    Rate (TPR) at True Negative Rate (TNR) equal to 1.
    
    The analysis tries to divide the normal and anomalous points just by
    placing a constant in a moving standard deviation space of the time
    series. It means that the time series is first projected into a moving
    standard deviation space with window of length `w` to be found, then a
    constant to divide the points is found. If the time series is
    multivariate, the constant will be a constant vector in which elements
    are the constants for the time series channels and each channel may be
    projected with a different window `w`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.
        
    y : array-like of shape (n_samples,)
        The labels of the time series.
        
    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.
        
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. It has three keys:
        `mov_std_score`, `upper_bound`, `lower_bound`, `window` and `diff_order`.
        The first is the moving standard deviation score which lies between 0
        and 1. The second is the upper bound for each channel that have been
        found to separate anomalies and normal points. The third is the lower
        bound for each channel. The fourth is the best window that has been
        found to yield the score. The last is the differencing that has been
        applied to the time series before searching for the moving standard
        deviation score. If any of the bounds is None it means that there is no
        separation keeping TNR at 1. When bounds are not None, a point is
        labelled anomalous if any feature of the moving average series of that
        point is greater or lower than found bounds.
    """
    result = _execute_movement_simplicity(x, y, diff, window_range, mov_std)
    result["mov_std_score"] = result["movement_score"]
    del result["movement_score"]
    return result


# TODO: continue to check until numba.typed.List and numba.typed.Dict will become stable and change
@jit(nopython=True, parallel=True)
def _fast_execute_mixed_score_simplicity(x,
                                         y,
                                         diff: int,
                                         windows_to_try: list[int]) -> tuple:
    """Computes the mixed simplicity score.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Same as `analyse_mixed_simplicity`.

    y : array-like of shape (n_samples,)
        Same as `analyse_mixed_simplicity`.

    diff : int, default=3
        Same as `analyse_mixed_simplicity`.

    windows_to_try : list[int]
        The windows to try to compute the movement simplicity score.

    Returns
    -------
    results : tuple
        A tuple with the results of the numba analyse constant simplicity as
        first elements, next the results of fast execute movement simplicity
        for avg, next the fast execute movement simplicity for std, then the
        mixed score.
    """
    const_res = _analyse_constant_simplicity(x, y, diff)
    mov_avg_res = _fast_execute_movement_simplicity(x, y, diff, windows_to_try, mov_avg)
    mov_std_res = _fast_execute_movement_simplicity(x, y, diff, windows_to_try, mov_std)
    
    mixed_score = 0
    if const_res[0] == mov_avg_res[0] and mov_avg_res[0] == mov_std_res[0] and const_res[0] == 0:
        # if all are 0 also mixed is 0
        mixed_score = 0
    else:
        # at least one of them is not 0
        if const_res[0] == 1 or mov_avg_res[0] == 1 or mov_std_res[0] == 1:
            # if at least one reaches perfection mixed is perfect
            mixed_score = 1
        else:
            # scores themselves are between 0 and 1, mixed must be computed
            mov_avg_input = x
            if mov_avg_res[-2] != 0:
                mov_avg_input = _diff_numpy(x, mov_avg_res[-2])
            mov_avg_series = mov_avg(mov_avg_input, mov_avg_res[-1])
            
            mov_std_input = x
            if mov_std_res[-2] != 0:
                mov_std_input = _diff_numpy(x, mov_std_res[-2])
            mov_std_series = mov_std(mov_std_input, mov_std_res[-1])
            
            pred = np.full(y.shape, False, dtype=np.bool_)
            for f in prange(x.shape[1]):
                # process constant labels
                if not math.isnan(const_res[1][f]):
                    pos = np.argwhere(x[:, f] >= const_res[1][f])
                    for e in pos:
                        pred[e[0]] = True
                if not math.isnan(const_res[2][f]):
                    pos = np.argwhere(x[:, f] <= const_res[2][f])
                    for e in pos:
                        pred[e[0]] = True
                
                # process moving average labels
                if not math.isnan(mov_avg_res[1][f]):
                    pos = np.argwhere(mov_avg_series[:, f] >= mov_avg_res[1][f])
                    pos = pos + mov_avg_res[-2]
                    for e in pos:
                        pred[e[0]] = True
                if not math.isnan(mov_avg_res[2][f]):
                    pos = np.argwhere(mov_avg_series[:, f] <= mov_avg_res[2][f])
                    pos = pos + mov_avg_res[-2]
                    for e in pos:
                        pred[e[0]] = True
                
                # process moving standard deviation labels
                if not math.isnan(mov_std_res[1][f]):
                    pos = np.argwhere(mov_std_series[:, f] >= mov_std_res[1][f])
                    pos = pos + mov_std_res[-2]
                    for e in pos:
                        pred[e[0]] = True
                if not math.isnan(mov_std_res[2][f]):
                    pos = np.argwhere(mov_std_series[:, f] <= mov_std_res[2][f])
                    pos = pos + mov_std_res[-2]
                    for e in pos:
                        pred[e[0]] = True
            
            mixed_score = _true_positive_rate(y, pred)
    
    return *const_res, *mov_avg_res, *mov_std_res, mixed_score


def analyse_mixed_simplicity(x,
                             y,
                             diff: int = 3,
                             window_range: tuple[int, int] | slice | list[int] = (2, 300)) -> dict:
    """Analyses if the time series is complexity mixing constant, moving average and moving standard deviation scores.
    
    A dataset may be constant simple, moving average simple or moving standard
    deviation simple. However, they may also be partially simple in each of
    these cases.
    
    A dataset is mixed simple if combining the previous three simplicity scores
    it is simple. The score mixed score is defined such that it is 1 when the
    dataset is mixed simple. It is 0 when no anomalies can be separated from the
    normal points without producing false positives. Therefore, the higher the
    score, the higher the number of anomalies that can be found just by placing
    a constant on the moving average series. The score is the True Positive Rate
    (TPR) at True Negative Rate (TNR) equal to 1.
    
    The analysis first compute the constant score, moving average score and
    standard deviation score. Once these scores are computed, they are mixed to
    produce the final set of discovered anomalies such that if an anomaly can be
    discovered by at least one approach, it is discovered also by the mixed
    approach. Therefore, the mixed score is always at least equal to the highest
    and always less than or equal to 1.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The time series to be analysed.
        
    y : array-like of shape (n_samples,)
        The labels of the time series.
        
    diff : int, default=3
        It is the maximum number of times the series might be differenced to
        find constant simplicity. If constant simplicity is found at
        differencing value of `i`, higher values will not be checked. If it
        is passed 0, the series will never be differenced.
        
    window_range : tuple[int, int] or slice or list[int], default=(2, 200)
        It is the range in which the window will be searched, the slice object
        describing the range and the step to be used to search windows or a list
        of windows to try. Theoretically, all window values (between 0 and the
        length of the time series) should be tried to verify whether a dataset
        is moving average simple. This parameter limits the search into a
        specific interval for efficiency reasons and because over certain window
        dimension may become a useless search.

    Returns
    -------
    analysis_result : dict
        Dictionary with the results of the analysis. The dictionary has the
        following keys: `mixed_score`, `const_result`, `mov_avg_result` and
        `mov_std_result`. The `mixed_score` key is the key of the mixed score of
        the analysed dataset. The `const_result` key is the key of the constant
        score analysis' results. The `mov_avg_result` key is the key of the
        moving average score analysis' results. The `mov_std_result` is the key
        of the moving standard deviation analysis' results.
    """
    x, y, windows_to_try = _check_analysis_inputs(x, y, diff, window_range)
    __module_logger.warning("currently typed.List and typed.Dict for numba are "
                            "still not used")
    results = _fast_execute_mixed_score_simplicity(x, y, diff, windows_to_try)
    
    const_score, const_upper, const_lower, const_diff = results[0], results[1], results[2], results[3]
    const_upper, const_lower = _fix_numba_upper_lower(const_upper, const_lower)
    
    avg_score, avg_upper, avg_lower, avg_diff, avg_window = results[4], results[5], results[6], results[7], results[8]
    avg_upper, avg_lower = _fix_numba_upper_lower(avg_upper, avg_lower)
    
    std_score, std_upper, std_lower, std_diff, std_window = results[9], results[10], results[11], results[12], results[13]
    std_upper, std_lower = _fix_numba_upper_lower(std_upper, std_lower)
    
    mixed_score = results[-1]
    
    const_result = {"constant_score": const_score, "upper_bound": const_upper, "lower_bound": const_lower, "diff_order": const_diff}
    avg_result = {"mov_avg_score": avg_score, "upper_bound": avg_upper, "lower_bound": avg_lower, "diff_order": avg_diff, "window": avg_window}
    std_result = {"mov_std_score": std_score, "upper_bound": std_upper, "lower_bound": std_lower, "diff_order": std_diff, "window": std_window}
    
    return {"mixed_score": mixed_score, "const_result": const_result, "mov_avg_result": avg_result, "mov_std_result": std_result}
