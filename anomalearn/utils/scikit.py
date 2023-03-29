import logging

import numpy as np


__module_logger = logging.getLogger(__name__)


def are_numpy_attr_equal(estimator1, estimator2, attributes: list[str]) -> bool:
    """Checks that the numpy attributes are equal in the two estimators.
    
    To be equal they must be both None or both instances of two equal numpy
    arrays.
    
    Parameters
    ----------
    estimator1
        The first scikit-learn estimator, or an object with numpy attributes.
    
    estimator2
        The second scikit-learn estimator, or an object with numpy attributes.
    
    attributes : list[str]
        The attributes to check.

    Returns
    -------
    are_attributes_equal : bool
        True when the attributes in the two estimators are equal.
    """
    for prop_name in attributes:
        try:
            this_prop = getattr(estimator1, prop_name)
            __module_logger.debug(f"estimator1.{prop_name}={this_prop}")
        except AttributeError:
            __module_logger.debug(f"estimator1 does not have {prop_name}")
            this_prop = None
        try:
            other_prop = getattr(estimator2, prop_name)
            __module_logger.debug(f"estimator2.{prop_name}={other_prop}")
        except AttributeError:
            __module_logger.debug(f"estimator2 does not have {prop_name}")
            other_prop = None
    
        # XOR between this_prop and other_prop being None
        if (this_prop is None) != (other_prop is None):
            return False
    
        if this_prop is not None and not np.array_equal(this_prop, other_prop):
            return False
        
    return True


def are_normal_attr_equal(estimator1, estimator2, attributes: list[str]) -> bool:
    """Checks that the normal attributes are equal in the two estimators.
    
    To be equal they must be both None or their equivalence must return True
    (namely, estimator1.attribute == estimator2.attribute is True).
    
    Parameters
    ----------
    estimator1
        The first scikit-learn estimator, or an object with numpy attributes.
    
    estimator2
        The second scikit-learn estimator, or an object with numpy attributes.
    
    attributes : list[str]
        The attributes to check.

    Returns
    -------
    are_attributes_equal : bool
        True when the attributes in the two estimators are equal.
    """
    for prop_name in attributes:
        this_prop = getattr(estimator1, prop_name)
        other_prop = getattr(estimator2, prop_name)
        __module_logger.debug(f"estimator1.{prop_name}={this_prop}")
        __module_logger.debug(f"estimator2.{prop_name}={other_prop}")
    
        if (this_prop is None) != (other_prop is None):
            return False

        if this_prop != other_prop:
            return False

    return True
