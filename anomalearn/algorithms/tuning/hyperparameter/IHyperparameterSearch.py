from abc import ABC
from typing import Callable
import abc

import numpy as np

from .. import ICrossValidation
from . import IHyperparameterSearchResults


class IHyperparameterSearch(ABC):
    """Interface describing hyperparameter searchers.
    """
    
    @abc.abstractmethod
    def search(self, x,
               y,
               objective_function: Callable[[np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             dict], float],
               cross_val_generator: ICrossValidation = None,
               train_test_data: bool = False,
               load_checkpoints: bool = False,
               *args,
               **kwargs) -> IHyperparameterSearchResults:
        """Search the best hyperparameter values.
        
        Results of the search must be saved to file for persistence. The format
        the results must have is to be an array of shape (tries + 1, params + 1).
        Basically, the search contains a header of strings containing the names
        of the parameters searched and the name of the performance measure used
        to assess the model's quality. The other rows are composed of the values
        of the model's parameters and its performance with those parameters on
        the given dataset. Moreover, the performance of the model must always
        be in the first column.
        
        If there already exists a history file, this function will prompt a
        message asking the user if he/she wants to overwrite the file. In case
        the user does not want to overwrite it, the search process is
        interrupted.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features) or list
            Array-like containing data over which to perform the search of the
            hyperparameters. If it is expressed as `list`, it represents the
            values of the training to pass to the objective function. In that
            case, it must have the same length of y. Each element of the list
            must be an iterable (such as `Tuple`) in which the first element is
            the training x and the second element is the testing x. The training
            set must be compatible with training labels, and testing set must
            be compatible with testing labels.
        
        y : array-like of shape (n_samples, n_target) or list
            Array-like containing targets of the data on which to perform the
            search of the hyperparameters. If it is expressed as `list` it
            represents the values of the training to pass to the objective
            function. In that case, it must have the same length of x. Each
            element of the list must be an iterable (such as `Tuple`) in which
            the first element are the training labels and the second element are
            the testing labels. The training labels must be compatible with
            training set, and testing labels must be compatible with testing set.
        
        objective_function : Callable
            It is a function training the model and evaluating its performances.
            The first argument is the training set, the second argument is
            the set of training labels, the third argument is the test set
            and the fourth argument is the set of test labels. The last is
            a dictionary of the parameters of the model. Basically,
            objective_function(train_data, train_labels, valid_data,
            valid_labels, parameters).
            
        cross_val_generator : ICrossValidation, default=None
            Only relevant if `train_test_data` is False. Otherwise, it is
            ignored. It is the cross validation generator returning a train/test
            generator. If nothing is passed, standard K Fold Cross validation is
            used with default scikit-learn parameters. It will be used on every
            configuration to obtain the k-fold score of the objective that is
            being evaluated for the configuration.
            
        train_test_data : bool, default=False
            A boolean stating if x and y are lists of iterables containing the
            splits to be used instead of "raw" data to be divided with any
            cross validation technique. The iterables must be of length two and
            must contain as first element the training set/labels and as second
            element the testing set/labels.
            
        load_checkpoints : bool, default=False
            A boolean stating if previous saved history should be loaded from
            checkpoint files. It True it will try to load the search history and
            if there is no checkpoint file, it will work as it was set to False.
        
        args
            Not used, present to allow multiple inheritance and signature change.
            It might be used by subclasses.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.
            It might be used by subclasses.

        Returns
        -------
        search_results : IHyperparameterSearchResults
            The results of the search.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_results(self) -> IHyperparameterSearchResults:
        """Get search results.
        
        Returns
        -------
        search_results : IHyperparameterSearchResults
            The results of the last search or the specified search at
            initialization.
        """
        raise NotImplementedError
