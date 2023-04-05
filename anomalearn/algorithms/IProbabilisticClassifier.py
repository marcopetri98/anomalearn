import abc

import numpy as np

from . import IClassifier


class IProbabilisticClassifier(IClassifier):
    """Interface identifying a machine learning probabilistic classifier.
    """
    
    @abc.abstractmethod
    def predict_proba(self, x, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
