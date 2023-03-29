import logging
import os
from typing import Callable

import numpy as np
from skopt.space import Categorical, Integer

from . import HyperparameterSearch
from .. import ICrossValidation


class TimeSeriesGridSearch(HyperparameterSearch):
    """Gird search over for time series datasets and models.
    """
    
    def __init__(self, parameter_space: list[Categorical | Integer],
                 saving_folder: str | os.PathLike,
                 saving_filename: str):
        super().__init__(parameter_space=parameter_space,
                         saving_folder=saving_folder,
                         saving_filename=saving_filename)

        self.__logger = logging.getLogger(__name__)
        self._tried_configs = {}

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
        # Input validation
        for parameter in self.parameter_space:
            if not (isinstance(parameter, Categorical) or isinstance(parameter, Integer)):
                raise ValueError("Cannot run grid search out of discrete values")
        
        self.__logger.info("Building the parameters grid")
        # Build dict of param:possible values
        space = dict()
        for parameter in self.parameter_space:
            if isinstance(parameter, Categorical):
                values = [category for category in parameter.categories]
            else:
                values = range(parameter.low, parameter.high + 1, 1)
            
            space[parameter.name] = values
        
        self.__logger.info("Starting the grid search")
        # Iterate over all possible configuration and call the objective
        sel_values = [0] * len(self.parameter_space)
        has_finished = False
        while not has_finished:
            config = [space[self.parameter_space[idx].name][sel_values[idx]]
                      for idx in range(len(sel_values))]
            
            # If the configuration has not been tried yet, objective is called
            _ = self._objective_call(x,
                                     y,
                                     objective_function,
                                     cross_val_generator,
                                     train_test_data,
                                     *config)
            
            # Change the configuration to try
            has_changed = False
            while not has_changed:
                for i in reversed(range(len(sel_values))):
                    sel_values[i] += 1
                    if sel_values[i] == len(space[self.parameter_space[i].name]) and i != 0:
                        sel_values[i] = 0
                    else:
                        has_changed = True
                        break
            
            # If all configuration has been tried, finish the loop
            if sel_values[0] == len(space[self.parameter_space[0].name]):
                has_finished = True
        
        self.__logger.info("The grid search ended correctly")
