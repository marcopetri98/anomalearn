import logging

from numba import jit
import numpy as np

from ..input_validation import check_array_1d


__module_logger = logging.getLogger(__name__)


def _check_binary_input(y_true,
                        y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Checks that the input is the result of binary classification.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    numpy_y_true : np.ndarray of shape (n_samples,)
        The numpy array of `y_true`.
    
    numpy_y_pred : np.ndarray of shape (n_samples,)
        The numpy array of `y_pred`.
        
    Raises
    ------
    ValueError
        If the arrays in input are not 1D, if they have more than 2 labels or
        if the labels are not 0 and 1.
    """
    check_array_1d(y_true, array_name="y_true")
    check_array_1d(y_pred, array_name="y_pred")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    __module_logger.debug(f"y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}")
    __module_logger.debug(f"np.unique(y_true)={np.unique(y_true.shape)}")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same shape")
    elif len(np.unique(y_true)) != 2:
        raise ValueError("binary confusion matrix needs exactly 2 labels in y_true")
    elif 1 not in np.unique(y_true) or 0 not in np.unique(y_true):
        raise ValueError("binary confusion matrix uses labels 0 and 1")
    
    return y_true, y_pred


@jit(nopython=True)
def _binary_confusion_matrix(y_true,
                             y_pred) -> tuple[int, int, int, int]:
    """Same as `binary_confusion_matrix`.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Same as `binary_confusion_matrix`.

    y_pred : array-like of shape (n_samples,)
        Same as `binary_confusion_matrix`.

    Returns
    -------
    tn : int
        True negatives.
    
    fp : int
        False positives.
    
    fn : int
        False negatives.
    
    tp : int
        True positives.
    """
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    
    for i in range(y_true.shape[0]):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
            
    return tn, fp, fn, tp


@jit(nopython=True)
def _true_positive_rate(y_true,
                        y_pred) -> float:
    """Same as `true_positive_rate`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Same as `true_positive_rate`.

    y_pred : array-like of shape (n_samples,)
        Same as `true_positive_rate`.

    Returns
    -------
    tpr : float
        Same as `true_positive_rate`.
    """
    _, _, fn, tp = _binary_confusion_matrix(y_true, y_pred)
    return tp / (tp + fn)


@jit(nopython=True)
def _true_negative_rate(y_true,
                        y_pred) -> float:
    """Same as `true_negative_rate`.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Same as `true_negative_rate`.
    
    y_pred : array-like of shape (n_samples,)
        Same as `true_negative_rate`.
        
    Returns
    -------
    tnr : float
        Same as `true_negative_rate`.
    """
    tn, fp, _, _ = _binary_confusion_matrix(y_true, y_pred)
    return tn / (tn + fp)


def binary_confusion_matrix(y_true,
                            y_pred) -> tuple[int, int, int, int]:
    """Gets the binary confusion matrix.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    tn : int
        True negatives.
    
    fp : int
        False positives.
    
    fn : int
        False negatives.
    
    tp : int
        True positives.
    """
    y_true, y_pred = _check_binary_input(y_true, y_pred)
    return _binary_confusion_matrix(y_true, y_pred)


def true_positive_rate(y_true,
                       y_pred) -> float:
    """Computes the True Positive Rate.
    
    This function is implemented in numba such that all the functions
    can call it.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    tpr : float
        The True Positive Rate.
    """
    y_true, y_pred = _check_binary_input(y_true, y_pred)
    return _true_positive_rate(y_true, y_pred)


def true_negative_rate(y_true,
                       y_pred) -> float:
    """Computes the True Negative Rate.
    
    This function is implemented in numba such that all the functions
    can call it.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The real labels for the points.
    
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the points.

    Returns
    -------
    tnr : float
        The True Negative Rate.
    """
    y_true, y_pred = _check_binary_input(y_true, y_pred)
    return _true_negative_rate(y_true, y_pred)
