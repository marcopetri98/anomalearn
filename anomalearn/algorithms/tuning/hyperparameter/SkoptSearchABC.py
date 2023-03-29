import abc
import logging
import os
from typing import Callable

import numpy as np
import skopt
from skopt.callbacks import CheckpointSaver
from skopt.space import Categorical, Integer, Real

from . import HyperparameterSearch, IHyperparameterSearchResults
from .. import ICrossValidation


class SkoptSearchABC(HyperparameterSearch):
    """HyperparameterSearch is a class used to search the hyperparameters.
    """
    __INVALID_SKOPT_KWARGS = ["func", "dimensions", "x0", "y0"]
    
    def __init__(self, parameter_space: list[Categorical | Integer | Real],
                 saving_folder: str | os.PathLike,
                 saving_filename: str):
        super().__init__(parameter_space=parameter_space,
                         saving_folder=saving_folder,
                         saving_filename=saving_filename)
        self.__logger = logging.getLogger(__name__)
        
        self.__x = None
        self.__y = None
        self.__train_test_data = False
        self.__cross_val_generator = None
        self.__minimized_objective = None

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
        """
        Parameters
        ----------
        kwargs
            It may have a keyword "skopt_kwargs" specifying the additional named
            arguments to pass to the skopt optimization function. It cannot have
            one of "func", "dimensions", "x0" or "y0" since they are specified
            by the wrapper to manage checkpoints and search.
        """
        return super().search(x=x,
                              y=y,
                              objective_function=objective_function,
                              train_test_data=train_test_data,
                              load_checkpoints=load_checkpoints,
                              *args,
                              **kwargs)

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
        """
        Parameters
        ----------
        kwargs
            It may have a keyword "skopt_kwargs" specifying the additional named
            arguments to pass to the skopt optimization function. It cannot have
            one of "func", "dimensions", "x0" or "y0" since they are specified
            by the wrapper to manage checkpoints and search.
        """
        # set parameters that can't be forwarded to the optimized function
        self.__minimized_objective = objective_function
        self.__x = x
        self.__y = y
        self.__cross_val_generator = cross_val_generator
        self.__train_test_data = train_test_data
        file_path = self.save_folder / (self.save_filename + ".pkl")
        
        checkpoint_saver = CheckpointSaver(file_path, compress=9)
        skopt_kwargs = dict()
        if "skopt_kwargs" in kwargs:
            skopt_kwargs = kwargs["skopt_kwargs"]
            for invalid_kw in self.__INVALID_SKOPT_KWARGS:
                if invalid_kw in skopt_kwargs:
                    self.__logger.info(f"keyword '{invalid_kw}' from "
                                       "skopt_kwargs has been removed since it "
                                       "is invalid")
                    del skopt_kwargs[invalid_kw]
        
        callbacks = [checkpoint_saver]
        if "callback" in skopt_kwargs.keys():
            callbacks.append(skopt_kwargs["callback"])
            del skopt_kwargs["callback"]
        
        if load_checkpoints and file_path.exists():
            previous_checkpoint = skopt.load(file_path)
            x0 = previous_checkpoint.x_iters
            y0 = previous_checkpoint.func_vals

            self._load_history()
            self._skopt_call(callbacks, skopt_kwargs, load_checkpoints, x0, y0)
        else:
            self._skopt_call(callbacks, skopt_kwargs, load_checkpoints)
    
    @abc.abstractmethod
    def _skopt_call(self, callbacks: list,
                    skopt_kwargs: dict,
                    load_checkpoints: bool,
                    x0=None,
                    y0=None) -> None:
        """Call the wrapped skopt optimization function.
        
        Parameters
        ----------
        callbacks : list
            It is the list of callbacks to be passed to the skopt optimization
            functions.
        
        skopt_kwargs : dict
            The dictionary of extra named arguments to pass to the skopt
            optimization function.
        
        load_checkpoints : bool
            A boolean stating if a previous checkpoint is being loaded.
        
        x0
            The initial input points loaded from the checkpoint.
        
        y0
            The evaluation of initial input points loaded from the checkpoint.

        Returns
        -------
        None
        """
        raise NotImplementedError
    
    def _skopt_objective(self, args: list) -> float:
        """Respond to a call with the parameters chosen by the Gaussian Process.
        
        Parameters
        ----------
        args : list
            The space parameters chosen by the gaussian process.

        Returns
        -------
        configuration_score : float
            The score of the configuration to be minimized.
        """
        score = self._objective_call(self.__x,
                                     self.__y,
                                     self.__minimized_objective,
                                     self.__cross_val_generator,
                                     self.__train_test_data,
                                     *args)
        return score
