import logging
import os

from skopt.space import Categorical, Integer, Real
import skopt

from . import SkoptSearchABC


class GaussianProcessesSearch(SkoptSearchABC):
    """Wrapper for the `gp_minimize` search of `skopt`.
    """
    
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
        self.__logger.info("Starting the gaussian processes minimization")
        
        if load_checkpoints:
            _ = skopt.gp_minimize(self._skopt_objective,
                                  self.parameter_space,
                                  x0=x0,
                                  y0=y0,
                                  callback=callbacks,
                                  **skopt_kwargs)
        else:
            _ = skopt.gp_minimize(self._skopt_objective,
                                  self.parameter_space,
                                  callback=callbacks,
                                  **skopt_kwargs)
