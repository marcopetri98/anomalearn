import unittest

import numpy as np

from anomalearn.algorithms.tuning.hyperparameter import \
    HyperparameterSearchResults


class TestHyperparameterSearchResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng()
        random_history = rng.random((10, 5)).tolist()
        header = [["Score", "Duration", "param1", "param2", "param3"]]
        cls.history = header + random_history
        
    def setUp(self) -> None:
        self.results = HyperparameterSearchResults(self.history)
        
    def test_creation(self):
        self.assertListEqual(self.results._history, self.history)
        
    def test_get_best_score(self):
        self.assertEqual(min([e[0] for e in self.history[1:]]), self.results.get_best_score())
    
    def test_get_best_config(self):
        best_config_i = [e[0] for e in self.history[1:]].index(min([e[0] for e in self.history[1:]]))
        best_config = dict()
        for i, key in enumerate(self.history[0]):
            best_config[key] = self.history[best_config_i + 1][i]
        self.assertDictEqual(best_config, self.results.get_best_config())
    
    def test_get_num_iterations(self):
        self.assertEqual(len(self.history) - 1, self.results.get_num_iterations())
    
    def test_get_history(self):
        self.assertListEqual(self.history, self.results.get_history())
