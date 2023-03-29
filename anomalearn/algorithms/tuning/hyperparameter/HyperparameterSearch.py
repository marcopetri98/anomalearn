import abc
import logging
import os
import time
from abc import ABC
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import check_X_y

from . import HyperparameterSearchResults, IHyperparameterSearch, IHyperparameterSearchResults
from .. import ICrossValidation
from .... import ObtainableABC
from ....utils import load_py_json, save_py_json


class HyperparameterSearch(IHyperparameterSearch, ABC):
    """Abstract class implementing some hyperparameter search methods.
    
    The abstract class implements basics functionality to run the optimization
    and to save results upon configuration tries. The checkpoints will be saved
    to a file called "NAME_history.checkpoint" where NAME is `saving_filename`.
    
    Parameters
    ----------
    parameter_space : list
        It is the space of the parameters to explore to perform hyperparameter
        search. This object must be a list of skopt space objects. To be precise,
        this function uses scikit-optimize as core library of implementation.

    saving_folder : str
        It is the folder where the search results must be saved.

    saving_filename : str
        It is the filename of the file where to save the results of the search.
        No extension should be inserted.
    """
    _DURATION = "Duration"
    _SCORE = "Score"
    _HISTORY = "_history.checkpoint"
    
    def __init__(self, parameter_space: list,
                 saving_folder: str | os.PathLike,
                 saving_filename: str):
        super().__init__()
        
        self.parameter_space = parameter_space
        self.save_folder = Path(saving_folder)
        self.save_filename = saving_filename
        
        self.__logger = logging.getLogger(__name__)
        self._search_history = None
    
    def _load_history(self):
        self._search_history = load_py_json(self.save_folder / (self.save_filename + self._HISTORY))
    
    def _save_history(self):
        save_py_json(self._search_history, self.save_folder / (self.save_filename + self._HISTORY))
    
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
        if not train_test_data:
            check_X_y(x, y)

            x = np.array(x)
            y = np.array(y)
        else:
            if not isinstance(x, list) or not isinstance(y, list):
                raise TypeError("x and y must be lists of the same length")
            else:
                new_x = []
                new_y = []
                
                for el1, el2 in zip(x, y):
                    if not isinstance(el1, ObtainableABC) or not isinstance(el2, ObtainableABC):
                        raise TypeError("x and y must contain iterables of "
                                        "dimension 2 implementing len and []")
                    elif len(el1) != 2 or len(el2) != 2:
                        raise ValueError("x or y contain iterables with length "
                                         "different from 2")
                    else:
                        check_X_y(el1[0], el2[0])
                        check_X_y(el1[1], el2[1])
                        
                        new_x.append((np.array(el1[0]), np.array(el1[1])))
                        new_y.append((np.array(el2[0]), np.array(el2[1])))
                        
                x = new_x
                y = new_y
        
        immediate_stop = False
        if load_checkpoints:
            self.__logger.info("Loading history from checkpoints")
            self._load_history()
        else:
            if (self.save_folder / (self.save_filename + self._HISTORY)).exists():
                response = input("There is an history file, do you want to "
                                 "overwrite it (you will lose it)? [y/n]: ")
                if response.lower() == "n" or response.lower() == "no":
                    immediate_stop = True
            
            self._search_history = None
        
        start_time = time.time()
        try:
            if immediate_stop:
                raise StopIteration
            
            self._run_optimization(x,
                                   y,
                                   objective_function,
                                   cross_val_generator,
                                   train_test_data,
                                   load_checkpoints,
                                   *args,
                                   **kwargs)
            
            self.__logger.info("The objective function has been optimized")
            self._save_history()
            final_history = self._create_result_history()
            
            self.__logger.info("Search history has been saved to file")
            results = HyperparameterSearchResults(final_history)
        except StopIteration:
            self.__logger.debug("Research has been interrupted")
            results = HyperparameterSearchResults([])
        
        end_time = time.time()
        self.__logger.info(f"Search ended and lasted for {end_time - start_time:.2f}s")
        
        return results
    
    def get_results(self) -> IHyperparameterSearchResults:
        self._load_history()
        history = self._create_result_history()
        return HyperparameterSearchResults(history)
    
    @abc.abstractmethod
    def _run_optimization(self, x,
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
                          **kwargs) -> None:
        """Runs the optimization search.
        
        This function implements the logic to explore the parameters and the
        learning method, e.g., a grid search will try all the parameters in the
        grid while a gaussian search will try to learn the distribution after
        several tries. This function must call multiple times `_objective_call`
        with the parameters it wants to try. This function will execute the
        objective function and will return the performance measure of the
        objective call with the given parameters.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features) or list
            The same as `search`, it must only be propagated to other methods.
        
        y : array-like of shape (n_samples, n_target) or list
            The same as `search`, it must only be propagated to other methods.
        
        objective_function : Callable
            The objective function to minimize.
            
        cross_val_generator : ICrossValidation, default=None
            The same as `search`, it must only be propagated to other methods.
            
        train_test_data : bool, default=False
            The same as `search`, it must only be propagated to other methods.
            
        load_checkpoints : bool, default=False
            The same as `search`, passed in case the inheriting class saves
            additional information for searches.
            
        Returns
        -------
        None
        
        Raises
        ------
        StopIteration
            If the search must be aborted due to any problem.
            
        Notes
        -----
        See :meth:`~tuning.hyperparameter.HyperparameterSearch.search` for more
        details about objective_function.
        """
        raise NotImplementedError
    
    def _objective_call(self, x,
                        y,
                        objective_function: Callable[[np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray,
                                                      dict], float],
                        cross_val_generator: ICrossValidation = None,
                        train_test_data: bool = False,
                        *args) -> float:
        """The function wrapping the loss to minimize.
        
        The function wraps the loss to minimize passed to the object by
        manipulating the dataset to obtain training and validation sets and by
        saving results to the search history.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features) or list
            The same as `search`.
        
        y : array-like of shape (n_samples, n_target) or list
            The same as `search`.
        
        objective_function : Callable
            The objective function to minimize.
            
        cross_val_generator : ICrossValidation, default=None
            The same as `search`.
            
        train_test_data : bool, default=False
            The same as `search`.
        
        args : list
            The list of all the parameters passed to the function to be able
            to run the algorithm.

        Returns
        -------
        function_value: float
            The value of the computed objective function.
            
        Notes
        -----
        See :meth:`~tuning.hyperparameter.HyperparameterSearch.search` for more
        details about objective_function.
        """
        raw_args = args
        params = self._build_input_dict(*args)
        self.__logger.info(f"The objective function is being evaluated with {params}")
        
        score = 0
        
        start_time = time.time()
        if not train_test_data:
            # The data are passed as numpy array and must be cross validated
            if cross_val_generator is None:
                cross_val_generator = KFold()
            
            for train, test in cross_val_generator.split(x, y):
                x_train, x_test = x[train], x[test]
                y_train, y_test = y[train], y[test]
                obj = objective_function(x_train, y_train, x_test, y_test, params)
                score += obj
            
            score /= cross_val_generator.get_n_splits()
        else:
            # The training and testing couples are directly passed
            for data, labels in zip(x, y):
                x_train, x_test = data[0], data[1]
                y_train, y_test = labels[0], labels[1]
                obj = objective_function(x_train, y_train, x_test, y_test, params)
                score += obj
                
            score /= len(x)
        end_time = time.time()
        self._add_search_entry(score, end_time - start_time, *raw_args)
        self._save_history()
        
        self.__logger.info(f"The tested configuration has a score of {score}")
        return score
    
    def _add_search_entry(self, score,
                          duration,
                          *args) -> None:
        """It adds an entry to the search history.
        
        Parameters
        ----------
        score : float
            The score of this configuration.
            
        duration : float
            The duration of trying this configuration.
        
        args
            The passed arguments to the optimization function.

        Returns
        -------
        None
        """
        params = self._build_input_dict(*args)
        
        # Since tuples are saved to json arrays and json arrays are always
        # converted to lists, I need to convert any tuple to a list to be able
        # to check the identity since results may be loaded from a checkpoint
        for key, item in params.items():
            if isinstance(item, tuple):
                params[key] = list(item)
        
        if self._search_history is None:
            self._search_history = {self._SCORE: [score],
                                    self._DURATION: [duration]}
            
            for key, value in params.items():
                self._search_history[key] = [value]
        else:
            self._search_history[self._SCORE].append(score)
            self._search_history[self._DURATION].append(duration)
            
            common_keys = set(self._search_history.keys()).intersection(set(params.keys()))
            only_history_keys = set(self._search_history.keys()).difference(set(params.keys())).difference({self._SCORE, self._DURATION})
            only_params_keys = set(params.keys()).difference(set(self._search_history.keys()))
            
            for key in common_keys:
                self._search_history[key].append(params[key])
            
            for key in only_history_keys:
                self._search_history[key].append(None)
            
            for key in only_params_keys:
                self._search_history[key] = [None] * len(self._search_history[self._SCORE])
                self._search_history[key][-1] = params[key]
    
    def _build_input_dict(self, *args) -> dict:
        """Build the dictionary of the parameters.

        Parameters
        ----------
        args
            The passed arguments to the optimization function.

        Returns
        -------
        parameters : dict[str]
            Dictionary with parameter names as keys and their values as values.
        """
        params = dict()
        for i in range(len(args)):
            param_name = self.parameter_space[i].name
            if param_name is None:
                param_name = f"parameter_{i}"
                
            params[param_name] = args[i]
        return params
    
    def _create_result_history(self) -> list:
        """Creates the search result list from history.
        
        Returns
        -------
        search : list
            The results of the search with as first row the names of the
            parameters of the model.
        """
        keys = [key
                for key in self._search_history.keys()]
        tries = [[self._search_history[key][i]
                  for key in self._search_history.keys()]
                 for i in range(len(self._search_history[self._SCORE]))]
        final_history = [keys]
        
        for try_ in tries:
            final_history.append(try_)
        
        return final_history
