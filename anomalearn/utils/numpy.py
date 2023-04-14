from numpy.random import RandomState
import numpy as np


def are_random_state_equal(rs1: RandomState, rs2: RandomState) -> bool:
    """Checks if the two random states are equal.

    Since the RandomState class in numpy does not override the __eq__ method of
    Python, the check using "==" will fail even if the internal state of the two
    objects is identical. Therefore, two random states are considered equal when
    their state is identical.

    Parameters
    ----------
    rs1 : RandomState
        The instance of the first random state.
    
    rs2 : RandomState
        The instance of the second random state.

    Returns
    -------
    are_random_state_equal : bool
        True if the internal state of the two instances is equal.

    Raises
    ------
    TypeError
        If the two objects are not instances of RandomState.
    """
    if not isinstance(rs1, RandomState) or not isinstance(rs2, RandomState):
        raise TypeError("rs1 and rs2 must be instances of RandomState")
    
    state1 = rs1.get_state(False)
    state2 = rs2.get_state(False)

    def are_equal(obj1, obj2) -> bool:
        if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
            if not np.array_equal(obj1, obj2, True):
                return False
        else:
            if obj1 != obj2:
                return False
            
        return True

    for key, item in state1.items():
        if key not in state2:
            return False
        
        other = state2[key]

        print("check types")
        print(type(item), item)
        print(type(other), other)

        if isinstance(item, dict) and isinstance(other, dict):
            for value1, value2 in zip(item.values(), other.values()):
                if not are_equal(value1, value2):
                    return False
        elif not are_equal(item, other):
            return False
    
    return True
