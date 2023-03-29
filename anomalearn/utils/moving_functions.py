import numpy as np
from numba import jit, prange


def mov_avg(x, window: int, clip: str = "right") -> np.ndarray:
    """Compute the moving average series of `x`.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The original time series.

    window : int
        The window dimension.

    clip : str, default="right"
        From which side to have one element less in case the window is even.

    Returns
    -------
    mov_avg : ndarray of shape (n_samples, n_features)
        The moving average time series with same shape as `x`.
    """
    if x.ndim == 1:
        x = x.reshape((-1, 1))
    x = np.ascontiguousarray(x, dtype=np.double)
    return _mov_avg(x, window, clip)


def mov_std(x, window: int, clip: str = "right") -> np.ndarray:
    """Compute the moving average series of `x`.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The original time series.

    window : int
        The window dimension.

    clip : str, default="right"
        From which side to have one element less in case the window is even.

    Returns
    -------
    mov_std : ndarray of shape (n_samples, n_features)
        The moving standard deviation time series with same shape as `x`.
    """
    if x.ndim == 1:
        x = x.reshape((-1, 1))
    x = np.ascontiguousarray(x, dtype=np.double)
    return _mov_std(x, window, clip)


@jit(nopython=True, parallel=True, fastmath=True)
def _mov_avg(x, window: int, clip: str = "right") -> np.ndarray:
    """Compute the moving average series of `x`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The original time series.

    window : int
        The window dimension.
        
    clip : str, default="right"
        From which side to have one element less in case the window is even.

    Returns
    -------
    mov_avg : ndarray of shape (n_samples, n_features)
        The moving average time series with same shape as `x`.
    """
    left = window // 2
    right = left - (window % 2 == 0)
    if clip == "left":
        left, right = right, left
    
    avg_series = np.zeros_like(x)
    for i in prange(x.shape[0]):
        start = i - left if i - left >= 0 else 0
        end = i + 1 + right
        for j in prange(x.shape[1]):
            avg_series[i, j] = np.nanmean(x[start:end, j])
    
    return avg_series


@jit(nopython=True, parallel=True, fastmath=True)
def _mov_std(x, window: int, clip: str = "right") -> np.ndarray:
    """Compute the moving average series of `x`.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The original time series.

    window : int
        The window dimension.
        
    clip : str, default="right"
        From which side to have one element less in case the window is even.

    Returns
    -------
    mov_std : ndarray of shape (n_samples, n_features)
        The moving standard deviation time series with same shape as `x`.
    """
    left = window // 2
    right = left - (window % 2 == 0)
    if clip == "left":
        left, right = right, left
    
    std_series = np.zeros_like(x)
    for i in prange(x.shape[0]):
        start = i - left if i - left >= 0 else 0
        end = i + 1 + right
        for j in prange(x.shape[1]):
            std_series[i, j] = np.nanstd(x[start:end, j])
    
    return std_series
