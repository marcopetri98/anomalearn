from abc import ABC
import abc


class IParametric(ABC):
    """Interface identifying a machine learning parametric model.
    
    Such a model has some parameters to be learned from data.
    """
    
    @abc.abstractmethod
    def fit(self, x, y=None, *args, **kwargs) -> None:
        """Fits the model to the given training data.
        
        Parameters
        ----------
        x : array-like
            The data used for fitting. Data must have at least two dimensions in
            which the first dimension represent the number of samples.

        y : array-like, default=None
            The target for the fitting data. Data must have at least two
            dimensions in which the first dimension represent the number of
            samples. Moreover, if not `None`, the first dimension must be the
            same as that of `x`.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        raise NotImplementedError
