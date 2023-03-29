import os
from typing import Callable

import numpy as np

from anomalearn.algorithms.tuning import ICrossValidation
from anomalearn.algorithms.tuning.hyperparameter import HyperparameterSearch


class HyperparameterSearchChild(HyperparameterSearch):
    def __init__(self, parameter_space: list,
                 saving_folder: str | os.PathLike,
                 saving_filename: str,
                 fake_values: list = None):
        super().__init__(parameter_space=parameter_space,
                         saving_folder=saving_folder,
                         saving_filename=saving_filename)
        
        self.fake_values = fake_values
    
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
        for fake_list in self.fake_values:
            self._objective_call(x, y, objective_function, cross_val_generator, train_test_data, *fake_list)
