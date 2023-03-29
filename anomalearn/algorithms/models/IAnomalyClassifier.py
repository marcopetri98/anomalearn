from abc import ABC

from ..IClassifier import IClassifier


class IAnomalyClassifier(IClassifier, ABC):
    """Interface identifying a machine learning anomaly classifier.
    """
