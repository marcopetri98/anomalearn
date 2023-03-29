import unittest
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.model_selection import KFold
from skopt.space import Integer, Categorical

from anomalearn.algorithms.tuning.hyperparameter import TimeSeriesGridSearch
from tests.anomalearn.algorithms.tuning.hyperparameter.TestHyperparameterSearch import \
    RandomScoreHolder


class TestTimeSeriesGridSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng()
        cls.series = rng.random((1000, 5), dtype=np.double)
        cls.labels = rng.integers(0, 1, 1000, dtype=np.intc, endpoint=True)
        cls.rng = rng
    
    def test_search_and_get_results(self):
        for low, high, categories in zip([0, 10, -10], [20, 50, 100], [[0, 1, 2, 3], [10, 20, 30], ["str1", "str2", "str3"]]):
            with TemporaryDirectory() as tmp_dir:
                scorer = RandomScoreHolder(self.rng, self.series, self.labels, KFold())
                
                tuner = TimeSeriesGridSearch([Integer(low, high),
                                              Categorical(categories, name="nice_category")],
                                             tmp_dir,
                                             "test")
    
                results = tuner.search(self.series, self.labels, scorer.random_score)
                
                configs = results.get_history()[1:]
                tried_param_0 = [e[-2] for e in configs]
                tried_param_1 = [e[-1] for e in configs]
                
                self.assertEqual(len(range(low, high + 1)) * len(categories), len(configs))
                self.assertSetEqual(set(range(low, high + 1)), set(tried_param_0))
                self.assertSetEqual(set(categories), set(tried_param_1))
