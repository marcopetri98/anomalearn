import numpy as np
from sklearn.utils import check_array


def get_rows_without_nan(x) -> np.ndarray:
    """Get the list of all the rows without nan values.
    
    The term rows refer to the first dimension of the array.
    
    Parameters
    ----------
    x : array-like
        It is an array like that may contain nan values.

    Returns
    -------
    rows_without_nan : ndarray
        An array containing the numbers of the rows without nan. Note that rows
        may have infinite values instead.
    """
    check_array(x, ensure_2d=False, allow_nd=True, force_all_finite=False)
    x = np.array(x)
    return np.argwhere(~np.isnan(np.sum(x.reshape((x.shape[0], -1)), axis=1))).squeeze()
