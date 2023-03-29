import logging
from typing import Iterable

import numpy as np


__module_logger = logging.getLogger(__name__)


def all_indices(list_: Iterable, arg) -> list[int]:
    """Finds all indices of `arg` in `list_`, if any.
    
    Parameters
    ----------
    list_ : Iterable
        It is a list in which we want to find occurrences of `arg`.
        
    arg : object
        It is the object we are looking for in `list_`.

    Returns
    -------
    indices : list of int
        It is the list containing all the indices of `list_` containing `arg`.
        If `arg` is not present in `list_`, an empty list will be returned.
    """
    __module_logger.debug(f"list_={list_}")
    indices = [idx
               for idx, elem in enumerate(list_) if elem == arg]
    return indices


def concat_list_array(array: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenates all the ndarray inside the iterable of numpy arrays.
    
    Parameters
    ----------
    array : Iterable[ndarray]
        A list of numpy arrays

    Returns
    -------
    array : ndarray
        The numpy array obtained by concatenation of all the arrays inside the
        list.
    """
    __module_logger.debug(f"array={array}")
    a_final = None
    for a in array:
        if a_final is None:
            a_final = a
        else:
            a_final = np.concatenate((a_final, a))
    return a_final
