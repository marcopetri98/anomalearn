from typing import Tuple

from sklearn.utils import check_array
import numpy as np


def check_array_general(x,
                        dimensions: int,
                        minimum_samples: Tuple = None,
                        force_all_finite: bool | str = True,
                        array_name: str = "x") -> None:
    """Checks that the `X` is an array-like with specified properties.
    
    Parameters
    ----------
    x
        The object to be controlled.
        
    dimensions : int
        The number of dimensions that the array must have.
    
    minimum_samples : Tuple, default=None
        The minimum number of elements for each dimension. If None, no minimum
        number of samples is checked.
    
    force_all_finite : bool or {"allow-nan"}, default=True
        If True all elements are forces to be finite values, infinity values and
        NaN values will imply an invalid array. With "allow-nan" the array can
        have finite values and NaN, but not infinity. With False the array can
        have any value.
    
    array_name : str, default="x"
        The name of the array to use in exceptions.

    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If `X` is not an array like or if it does not satisfy all properties.
    """
    if minimum_samples is not None and len(minimum_samples) != dimensions:
        raise ValueError("minimum_samples must have at least shape elements, or"
                         " it can be None")

    if dimensions == 2:
        if minimum_samples is not None:
            try:
                check_array(x,
                            ensure_min_samples=minimum_samples[0],
                            ensure_min_features=minimum_samples[1],
                            force_all_finite=force_all_finite)
            except Exception as e:
                raise ValueError("x is not an array-like with desired features") from e
        else:
            try:
                check_array(x, force_all_finite=force_all_finite)
            except Exception as e:
                raise ValueError("x is not an array-like with desired features") from e
    else:
        try:
            check_array(x, ensure_2d=False, allow_nd=True, force_all_finite=force_all_finite)
        except Exception as e:
            raise ValueError("x is not an array-like with desired features") from e
        
        np_arr = np.array(x)
        
        if len(np_arr.shape) != dimensions:
            raise ValueError(array_name + " doesn't have the specified shape")
        
        if minimum_samples is not None:
            for i, dim in enumerate(np_arr.shape):
                if dim < minimum_samples[i]:
                    raise ValueError(array_name + " have too few elements "
                                                  " at dimension " + str(i))


def check_array_1d(x, force_all_finite: bool | str = True, array_name: str = "x") -> None:
    """Checks that `X` is an array-like with one dimension.
    
    Parameters
    ----------
    x
        An object to be tested for array-like interface.
    
    force_all_finite : bool or {"allow-nan"}, default=True
        If True all elements are forces to be finite values, infinity values and
        NaN values will imply an invalid array. With "allow-nan" the array can
        have finite values and NaN, but not infinity. With False the array can
        have any value.
    
    array_name : str, default="x""
        The name of the array to use in exceptions.

    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the object is not an array-like with one dimension.
    """
    try:
        check_array(x, ensure_2d=False, force_all_finite=force_all_finite)
    except Exception as e:
        raise ValueError("x is not an array-like with desired features") from e
    
    x = np.array(x)
    
    if x.ndim > 1:
        raise ValueError(array_name + " must be 1 dimensional array")


def check_x_y_smaller_1d(x, y, force_all_finite: bool | str = True, x_name: str = "x", y_name: str = "y"):
    """Checks that `X` has at most as many elements as `y` and that both are 1d.
    
    Parameters
    ----------
    x
        An object to be tested for array-like interface.
        
    y
        An object to be tested for array-like interface.
    
    force_all_finite : bool or {"allow-nan"}, default=True
        If True all elements are forces to be finite values, infinity values and
        NaN values will imply an invalid array. With "allow-nan" the array can
        have finite values and NaN, but not infinity. With False the array can
        have any value.
    
    x_name : str, default=None
        The name of the `X` array to use in exceptions.
        
    y_name : str, default=None
        The name of the `y` array to use in exceptions.

    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the object is not an array-like with one dimension.
    """
    try:
        check_array_1d(x, array_name=x_name, force_all_finite=force_all_finite)
        check_array_1d(y, array_name=y_name, force_all_finite=force_all_finite)
    except Exception as e:
        raise ValueError("x is not an array-like with desired features") from e
    
    x = np.array(x)
    y = np.array(y)
    
    if y.size < x.size:
        raise ValueError(x_name + " cannot have more elements than " + y_name)
