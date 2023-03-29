import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.model_selection import KFold
from skopt.space import Categorical, Real, Integer

from tests.anomalearn.algorithms.tuning.hyperparameter.TestHyperparameterSearch import \
    RandomScoreHolder
from tests.anomalearn.algorithms.tuning.hyperparameter.stubs.SkoptSearchABCChild import \
    SkoptSearchABCChild


class TestSkoptSearchABC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng()
        cls.series = rng.random((1000, 5), dtype=np.double)
        cls.labels = rng.integers(0, 1, 1000, dtype=np.intc, endpoint=True)
        cls.rng = rng
    
    def test_search_and_get_results(self):
        with TemporaryDirectory() as tmp_dir:
            scorer = RandomScoreHolder(np.random.default_rng(100), self.series, self.labels, KFold())
            fake_values = [[0, 0, 0], [1, 0, 0], [0, 10, 0], [0.5, 0, 0]]
                            
            space = [Integer(0, 100), Categorical([0, 10, 20, 30]), Real(0, 100)]
            tuner = SkoptSearchABCChild(space, Path(tmp_dir), "test", fake_values=fake_values)
            
            # tests are inside the stub because it should test that the
            # skopt call is called with the right arguments both in case of
            # loading a previous history and in case of creating a new one
            result = tuner.search(self.series, self.labels, scorer.random_score, skopt_kwargs={"test": "value", "tester": self})
            
            self.assertTrue((Path(tmp_dir) / "test.pkl"))
            y0 = [e[0] for e in result.get_history()[1:]]

            scorer = RandomScoreHolder(np.random.default_rng(100), self.series, self.labels, KFold())
            tuner = SkoptSearchABCChild(space, Path(tmp_dir), "test", fake_values=fake_values, test_loading=True, previous_y=y0)
            _ = tuner.search(self.series, self.labels, scorer.random_score, load_checkpoints=True, skopt_kwargs={"test": "value", "tester": self})
