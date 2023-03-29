import os

import numpy as np
import scipy
from skopt.callbacks import CheckpointSaver
from skopt.space import Categorical, Integer, Real

from anomalearn.algorithms.tuning.hyperparameter import SkoptSearchABC


class SkoptSearchABCChild(SkoptSearchABC):
    def __init__(self, parameter_space: list[Categorical | Integer | Real],
                 saving_folder: str | os.PathLike,
                 saving_filename: str,
                 fake_values: list = None,
                 test_loading: bool = False,
                 previous_y=None):
        super().__init__(parameter_space=parameter_space,
                         saving_folder=saving_folder,
                         saving_filename=saving_filename)

        self.fake_values = fake_values
        self.test_loading = test_loading
        self.previous_y = previous_y
        
        self.x_iters = []
        self.func_vals = []
        
    def _skopt_call(self, callbacks: list,
                    skopt_kwargs: dict,
                    load_checkpoints: bool,
                    x0=None,
                    y0=None) -> None:
        tester = skopt_kwargs["tester"]
        tester.assertEqual("value", skopt_kwargs["test"])
        skopt_saver: CheckpointSaver = callbacks[0]
        
        tester.assertIsInstance(skopt_saver, CheckpointSaver)
        
        if not self.test_loading:
            tester.assertIsNone(x0)
            tester.assertIsNone(y0)
        else:
            tester.assertIsNotNone(x0)
            tester.assertIsNotNone(y0)
            
            tester.assertListEqual(self.fake_values, x0)
            tester.assertListEqual(self.previous_y, y0)
        
        for fake_list in self.fake_values:
            score = self._skopt_objective(fake_list)
            res = scipy.optimize.minimize(lambda x: x[0]*x[1]*x[2], np.array([10, 10, 10]))
            
            self.x_iters.append(fake_list)
            self.func_vals.append(score)
            res["x_iters"] = self.x_iters
            res["func_vals"] = self.func_vals
            
            skopt_saver(res)
