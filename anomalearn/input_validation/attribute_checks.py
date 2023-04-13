def is_var_of_type(variable,
                   allowed_types: list) -> bool:
    """Checks if a variable is of at least one of the specified types
    
    Parameters
    ----------
    variable
        The variable over which the type must be checked.
    
    allowed_types : list
        The list of the allowed type for the variable. Eventually including None.

    Returns
    -------
    is_type_ok : bool
        True when the variable has an allowed type, False otherwise.
    """
    for type_ in allowed_types:
        if variable is None and type_ is None:
            return True
        elif isinstance(variable, type_):
            return True
        
    return False


def check_attributes_exist(estimator,
                           attributes: str | list[str]) -> None:
    """Checks if the attributes are defined in estimator.

    Parameters
    ----------
    estimator : object
        The estimator on which we want to verify if the attribute is set.

    attributes : str or list of str
        The attributes to check the existence in the estimator.

    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If at least one of the attributes is not defined in the estimator.
    """
    if isinstance(attributes, str):
        if attributes not in estimator.__dict__.keys():
            raise ValueError(f"{estimator.__class__} does not have attribute "
                             f"{attributes}")
    else:
        for attribute in attributes:
            if attribute not in estimator.__dict__.keys():
                raise ValueError(f"{estimator.__class__} does not have "
                                 f"attribute {attribute}")


def check_not_default_attributes(estimator,
                                 attributes: dict,
                                 error: str = "Attributes have default values") -> None:
    """Checks if the attributes have the default not trained value.
    
    It raises an exception if at least one of the attribute has the default not
    trained value.
    
    Parameters
    ----------
    estimator : object
        The estimator on which we want to verify if the attribute is set.
        
    attributes : dict
        Keys are string representing the attributes' names and the values are
        the standard values used when the estimator has not been trained yet.

    error : str
        The error string that is thrown.

    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If at least one of the specified attributes has the default value.
    """
    for key, value in attributes.items():
        check_attributes_exist(estimator, key)
        attr_val = getattr(estimator, key)
        if value is None:
            if attr_val is None:
                raise ValueError(error)
        else:
            if attr_val == value:
                raise ValueError(error)


def check_argument_types(arguments: list,
                         expected_types: list,
                         names: list = None) -> None:
    """Checks whether the arguments have the correct type.
    
    Parameters
    ----------
    arguments : list
        The arguments to check.
    
    expected_types : list
        The expected types for the arguments, if multiple values are accepted,
        the ith element of this list is a list. The element at ith position is
        the expected type for the argument at ith position.
        
    names : list, default=None
        The names of the arguments to print in the error.

    Returns
    -------
    None
    
    Raises
    ------
    TypeError
        If arguments or expected_types are not lists or if any of the arguments
        has wrong type.
        
    ValueError
        If the lists have different lengths.
    """
    # check type errors
    if not isinstance(arguments, list):
        raise TypeError("arguments must be a list")
    if not isinstance(expected_types, list):
        raise TypeError("expected_types must be a list")
    if names is not None and not isinstance(names, list):
        raise TypeError("names must be a list")
    
    # check value errors
    if len(arguments) != len(expected_types):
        raise ValueError("arguments and expected_types must have the same length")
    if names is not None and len(arguments) != len(names):
        raise ValueError("arguments and names must have the same length")
    
    # implementation
    for i, arg in enumerate(arguments):
        wrong_type = True
        if isinstance(expected_types[i], list):
            for arg_type in expected_types[i]:
                if arg_type is not None:
                    if isinstance(arg, arg_type):
                        wrong_type = False
                        break
                else:
                    if arg is None:
                        wrong_type = False
        else:
            if expected_types[i] is not None:
                if isinstance(arg, expected_types[i]):
                    wrong_type = False
            else:
                if arg is None:
                    wrong_type = False
        
        if wrong_type:
            if names is not None:
                raise TypeError(f"{names[i]} must be of type {expected_types[i]}")
            else:
                raise TypeError(f"{arg} should be of type {expected_types[i]}")
