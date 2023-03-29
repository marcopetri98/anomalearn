import abc

from . import IParametric


class IMultipleParametric(IParametric):
    """Interface identifying a machine learning parametric model with multiple fit.
    
    This class implements extends the standard IParametric interface to describe
    a model that can be trained on multiple datasets at the same time. E.g.,
    if we want to train on two different datasets the model we should use the
    function :meth:`models.IMultipleParametric.IMultipleParametric.fit_multiple`.
    """
    
    @abc.abstractmethod
    def fit_multiple(self, x: list, y: list = None, *args, **kwargs) -> None:
        """Fits the model to the given training data.
        
        Parameters
        ----------
        x : list
            The list of the fitting data to be used. The elements of the list
            must respect the requirements of :meth:`algorithms.IParametric.fit`.

        y : list, default=None
            The target for the fitting data. The list must have the same
            dimension as `x`. Moreover, `y` elements are pairwise constrained to
            `x` elements as stated in :meth:`algorithms.IParametric.fit`.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        None
        """
        raise NotImplementedError
