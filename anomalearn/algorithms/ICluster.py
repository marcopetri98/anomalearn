import abc
from abc import ABC

import numpy as np


class ICluster(ABC):
    """Interface identifying a machine learning algorithm performing clustering.
    """
    
    @abc.abstractmethod
    def cluster(self, x, *args, **kwargs) -> np.ndarray:
        """Clusters the data.
        
        Parameters
        ----------
        x : array-like
            The data used for clustering. Data must have at least two dimensions
            in which the first dimension represent the number of samples.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        clusters : ndarray of shape (n_samples, n_clusters)
            An array identifying the cluster at which each point is associated.
        """
        raise NotImplementedError
