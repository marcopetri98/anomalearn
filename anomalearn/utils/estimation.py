import logging
from typing import Tuple

import numpy as np
from numpy.linalg import LinAlgError
from sklearn.utils import check_array


__module_logger = logging.getLogger(__name__)


def estimate_mean_covariance(x) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates mean and covariance matrix.

    The standard deviation or covariance matrix use ddof=1 for the estimation.
    If the standard deviation is 0 or the sum of the elements in the covariance
    matrix is 0, the quantity 1e-10 is added to it. The function is capable of
    estimating the mean and the covariance matrix also in presence of some NaN
    values, given that there are enough numbers for each dimension.

    Parameters
    ----------
    x : array-like of shape (n,m)
        The vectors over which the mean and covariance matrix must be computed.
        The observations are on the first dimension and the features on the
        second dimension. Thus, it is an array of `n` vectors with `m` features.

    Returns
    -------
    mean : ndarray
        The mean vector.

    cov : ndarray
        The covariance matrix between the features.

    inv_cov : ndarray
        The inverse of the covariance matrix.
    """
    check_array(x, force_all_finite="allow-nan")
    errors = np.ma.array(x, mask=np.isnan(x))
    __module_logger.info("errors have been successfully converted to masked array")

    mean = np.ma.mean(errors, axis=0)
    cov = np.ma.cov(errors, rowvar=False, ddof=1) if errors.ndim != 1 and errors.shape[1] != 1 else np.ma.std(errors, axis=0, ddof=1)
    is_vector = errors.ndim != 1 and errors.shape[1] != 1
    __module_logger.info("mean and covariance/standard deviation computed")

    if (errors.ndim != 1 and errors.shape[1] != 1) and np.any(np.linalg.eigvals(cov) < 0):
        __module_logger.critical("covariance matrix has strictly negative eigenvalues")
        raise ValueError("Impossible to compute the covariance matrix.")

    if np.sum(cov) == 0:
        __module_logger.debug(f"np.sum(cov)={np.sum(cov)}")
        cov += 1e-10

    if is_vector:
        try:
            inv_cov = np.linalg.inv(cov)
        except LinAlgError:
            __module_logger.info("computing pseudo-inverse of covariance matrix")
            inv_cov = np.linalg.pinv(cov)
    else:
        inv_cov = 1 / cov

    return mean, cov, inv_cov
