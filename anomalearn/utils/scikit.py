from typing import Any
import logging

from sklearn.tree._tree import Tree
import numpy as np

from . import are_random_state_equal


__module_logger = logging.getLogger(__name__)


def _get_property(estimator, prop_name: str, estimator_name: str) -> Any:
    """Extracts the property from the estimator if it exist.
    
    Parameters
    ----------
    estimator
        The scikit-learn estimator, or an object.
    
    prop_name : str
        The property to extract.
    
    estimator_name : str
        The name of the estimator to use in debug messages.

    Returns
    -------
    property
        The value of the property if it exists, otherwise None.
    """
    try:
        this_prop = getattr(estimator, prop_name)
        __module_logger.debug(f"{estimator_name}.{prop_name}={this_prop}")
    except AttributeError:
        __module_logger.debug(f"{estimator_name} does not have {prop_name}")
        this_prop = None
        
    return this_prop


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
        this_prop = _get_property(estimator1, prop_name, "estimator1")
        other_prop = _get_property(estimator2, prop_name, "estimator2")
    
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
        this_prop = _get_property(estimator1, prop_name, "estimator1")
        other_prop = _get_property(estimator2, prop_name, "estimator2")
    
        if (this_prop is None) != (other_prop is None):
            return False

        if this_prop != other_prop:
            return False

    return True


def are_list_attr_equal(estimator1, estimator2, attributes: list[str]) -> bool:
    """Checks that the attributes are equal lists in the two estimators.
    
    To be equal they must be both None or their equivalence must return True
    (namely, estimator1.attribute == estimator2.attribute is True), if the
    elements of the lists are numpy arrays the numpy.array_equal function is
    used, otherwise python equality operator is used.
    
    Parameters
    ----------
    estimator1
        The first scikit-learn estimator, or an object with list attributes.
    
    estimator2
        The second scikit-learn estimator, or an object with list attributes.
    
    attributes : list[str]
        The attributes to check.

    Returns
    -------
    are_attributes_equal : bool
        True when the attributes in the two estimators are equal.
    """
    for prop_name in attributes:
        this_prop = _get_property(estimator1, prop_name, "estimator1")
        other_prop = _get_property(estimator2, prop_name, "estimator2")
    
        if (this_prop is None) != (other_prop is None):
            return False

        if this_prop is not None:
            if len(this_prop) != len(other_prop):
                return False
            
            for el1, el2 in zip(this_prop, other_prop):
                if not isinstance(el1, np.ndarray) and el1 != el2:
                    return False
                
                if isinstance(el1, np.ndarray) and not np.array_equal(el1, el2):
                    return False

    return True


def are_random_state_attr_equal(estimator1, estimator2, attributes: list[str]) -> bool:
    """Checks that the two random states are equal in the two estimators.

    Parameters
    ----------
    estimator1
        The first scikit-learn estimator, or an object with list attributes.
    
    estimator2
        The second scikit-learn estimator, or an object with list attributes.
    
    attributes : list[str]
        The attributes to check.

    Returns
    -------
    are_random_state_equal : bool
        True when the random state attributes in the two estimators are equal.
    """
    for prop_name in attributes:
        this_prop = _get_property(estimator1, prop_name, "estimator1")
        other_prop = _get_property(estimator2, prop_name, "estimator2")
    
        if (this_prop is None) != (other_prop is None):
            return False
        
        if this_prop is not None:
            if not isinstance(this_prop, type(other_prop)):
                return False

            if isinstance(this_prop, int) and this_prop != other_prop:
                return False
            if isinstance(this_prop, np.random.RandomState) and not are_random_state_equal(this_prop, other_prop):
                return False
            
    return True


def are_tree_attr_equal(estimator1, estimator2, attributes: list[str]) -> bool:
    """Checks that the tree attributes are equal in the two estimators.
    
    To be equal they must be both None or all of their fields must be the same.
    This function checks whether both are instances of the class defined in
    `sklearn.tree._tree` and are identical, i.e., their attributes contain the
    same values.
    
    Parameters
    ----------
    estimator1
        The first scikit-learn estimator, or an object with list attributes.
    
    estimator2
        The second scikit-learn estimator, or an object with list attributes.
    
    attributes : list[str]
        The attributes to check.

    Returns
    -------
    are_attributes_equal : bool
        True when the tree attributes in the two estimators are equal.

    Notes
    -----
    Observe that the documentation for this function is not available on the
    `scikit-learn` website, you can obtain the documentation of this class by
    using the command help(sklearn.tree._tree.Tree) as pointed out in the
    `scikit-learn` documentation.

    Raises
    ------
    TypeError
        If the attributes does not contain instances of trees.
    """
    for prop_name in attributes:
        this_prop = _get_property(estimator1, prop_name, "estimator1")
        other_prop = _get_property(estimator2, prop_name, "estimator2")
    
        if (this_prop is None) != (other_prop is None):
            return False
        
        if this_prop is not None:
            if not isinstance(this_prop, type(other_prop)):
                return False
            
            if not isinstance(this_prop, Tree):
                raise TypeError("This function checks Tree, found a variable of"
                                f" type {type(this_prop)}")
            
            normal_attrs = ["node_count", "capacity", "max_depth"]
            numpy_attrs = ["children_left", "children_right", "feature",
                           "threshold", "value", "impurity", "n_node_samples",
                           "weighted_n_node_samples"]
            
            if not are_normal_attr_equal(this_prop, other_prop, normal_attrs):
                return False
            if not are_numpy_attr_equal(this_prop, other_prop, numpy_attrs):
                return False

    return True
