import logging
import os

import skopt
from skopt.space import Categorical, Integer, Real

from . import SkoptSearchABC


class ForestSearch(SkoptSearchABC):
    def __init__(self, parameter_space: list[Categorical | Integer | Real],
                 saving_folder: str | os.PathLike,
                 saving_filename: str):
        super().__init__(parameter_space=parameter_space,
                         saving_folder=saving_folder,
                         saving_filename=saving_filename)
        self.__logger = logging.getLogger(__name__)
    
    def _skopt_call(self, callbacks: list,
                    skopt_kwargs: dict,
                    load_checkpoints: bool,
                    x0=None,
                    y0=None) -> None:
        self.__logger.info("Starting the tree based regression minimization")
        
        if load_checkpoints:
            _ = skopt.forest_minimize(self._skopt_objective,
                                      self.parameter_space,
                                      x0=x0,
                                      y0=y0,
                                      callback=callbacks,
                                      **skopt_kwargs)
        else:
            _ = skopt.forest_minimize(self._skopt_objective,
                                      self.parameter_space,
                                      callback=callbacks,
                                      **skopt_kwargs)
