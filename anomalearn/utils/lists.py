from typing import Iterable
import logging

import numpy as np


__module_logger = logging.getLogger(__name__)


def all_indices(sequence: Iterable, arg) -> list[int]:
    """Finds all indices of `arg` in `sequence`, if any.
    
    Parameters
    ----------
    sequence : Iterable
        It is a list in which we want to find occurrences of `arg`.
        
    arg : object
        It is the object we are looking for in `sequence`.

    Returns
    -------
    indices : list of int
        It is the list containing all the indices of `sequence` containing `arg`.
        If `arg` is not present in `sequence`, an empty list will be returned.
    """
    __module_logger.debug(f"sequence={sequence}")
    indices = [idx
               for idx, elem in enumerate(sequence) if elem == arg]
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
