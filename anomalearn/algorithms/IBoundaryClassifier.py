import abc

import numpy as np

from . import IClassifier


class IBoundaryClassifier(IClassifier):
    """Interface identifying a machine learning classifier computing a boundary.
    """
    
    @abc.abstractmethod
    def decision_function(self, x, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
