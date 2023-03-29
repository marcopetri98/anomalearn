import abc

import numpy as np


class IAnomalyScorer(object):
    """Interface identifying a machine learning algorithm giving anomaly scores.
    """

    @abc.abstractmethod
    def anomaly_score(self, x, *args, **kwargs) -> np.ndarray:
        """Computes the anomaly score of the given points.

        Scores in the range [0,1], the higher is the score the more abnormal
        the point is. **Please, note that** if the model is parametric (inherits
        from IParametric) you must first perform fit on training data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The points for which we must compute the anomaly score.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        anomaly_scores : ndarray of shape (n_samples,)
            The scores of the points.
        """
        pass
