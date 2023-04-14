from abc import ABC
import abc

import numpy as np


class IAnomalyScorer(ABC):
    """Interface identifying a machine learning algorithm giving anomaly scores.
    """

    @abc.abstractmethod
    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        """Computes the anomaly score of the given points.

        The higher is the score the more abnormal the point is.

        Parameters
        ----------
        x : array-like
            The data to be scored. Data must have at least two dimensions in
            which the first dimension represent the number of samples.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        anomaly_scores : ndarray of shape (n_samples, 1)
            The labels resulted from the scoring where "n_samples" is the number
            of samples of `x`.
        """
        raise NotImplementedError
