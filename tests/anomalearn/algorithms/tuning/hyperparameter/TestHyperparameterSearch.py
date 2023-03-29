import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.model_selection import KFold
from skopt.space import Integer, Categorical, Real

from tests.anomalearn.algorithms.tuning.hyperparameter.stubs.HyperparameterSearchChild import \
    HyperparameterSearchChild


class RandomScoreHolder(object):
    def __init__(self, rng,
                 series,
                 labels,
                 splitter):
        super().__init__()
        
        self.rng = rng
        self.series = series
        self.labels = labels
        
        self.n_calls = 0
        self.generated_scores = []
        self.splitter = splitter
        self.tried_params = []
        
    def random_score(self, x_train, y_train, x_test, y_test, params):
        for i, (train, test) in enumerate(self.splitter.split(self.series, self.labels)):
            if i == self.n_calls:
                np.testing.assert_array_equal(self.series[train], x_train)
                np.testing.assert_array_equal(self.labels[train], y_train)
                np.testing.assert_array_equal(self.series[test], x_test)
                np.testing.assert_array_equal(self.labels[test], y_test)
                self.n_calls += 1
                break
                
        if self.splitter.get_n_splits() == self.n_calls:
            self.n_calls = 0
            
        rand_score = self.rng.random(1, dtype=np.double)
        self.generated_scores.append(rand_score[0])
        self.tried_params.append(params)
        return rand_score[0]


class TestHyperparameterSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng()
        cls.series = rng.random((1000, 5), dtype=np.double)
        cls.labels = rng.integers(0, 1, 1000, dtype=np.intc, endpoint=True)
        cls.rng = rng
    
    def test_search_and_get_results(self):
        with TemporaryDirectory() as tmp_dir:
            for split_i, splitter in enumerate([KFold(), KFold(n_splits=10), KFold(), KFold()]):
                scorer = RandomScoreHolder(self.rng, self.series, self.labels, splitter)
                
                tmp_dir_obj = Path(tmp_dir)
                if split_i in [0, 1, 2]:
                    for file in tmp_dir_obj.glob("*"):
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            file.rmdir()
            
                fake_values = [[0, 0, 0], [1, 0, 0], [0, 10, 0], [0.5, 0, 0]]
                multiplier = 1
                
                tuner = HyperparameterSearchChild([Integer(0, 100),
                                                   Categorical([0, 10, 20, 30], name="Galadriel"),
                                                   Real(0, 100)],
                                                  Path(tmp_dir),
                                                  "test",
                                                  fake_values=fake_values)
                
                if split_i == 0:
                    results = tuner.search(self.series, self.labels, scorer.random_score, None, False, False)
                elif split_i == 1:
                    results = tuner.search(self.series, self.labels, scorer.random_score, splitter, False, False)
                elif split_i == 2:
                    x_couples = []
                    y_couples = []
                    for train, test in splitter.split(self.series, self.labels):
                        x_couples.append((np.array(self.series[train]), np.array(self.series[test])))
                        y_couples.append((np.array(self.labels[train]), np.array(self.labels[test])))
                    results = tuner.search(x_couples, y_couples, scorer.random_score, None, True, False)
                else:
                    multiplier = 2
                    results = tuner.search(self.series, self.labels, scorer.random_score, None, False, True)
                    
                got_results = tuner.get_results()
                history_path = tmp_dir_obj / "test_history.checkpoint"
                
                generated_scores = scorer.generated_scores
                tried_params = scorer.tried_params
                
                self.assertTrue(history_path.exists())
                self.assertEqual(len(fake_values) * splitter.get_n_splits(), len(generated_scores))
                self.assertEqual(len(fake_values) * splitter.get_n_splits(), len(tried_params))
                
                # assert that results after search and get results are the same
                self.assertDictEqual(results.get_best_config(), got_results.get_best_config())
                self.assertEqual(results.get_best_score(), got_results.get_best_score())
                self.assertEqual(results.get_num_iterations(), got_results.get_num_iterations())
                self.assertListEqual(results.get_history(), got_results.get_history())
                
                if multiplier == 2:
                    for i in range(len(previous_results.get_history())):
                        self.assertListEqual(previous_results.get_history()[i], results.get_history()[i])
                
                final_scores = []
                final_params = []
                # reduce the generated lists
                for i in range(len(fake_values)):
                    start = i * splitter.get_n_splits()
                    end = start + splitter.get_n_splits()
                    final_scores.append(sum(generated_scores[start:end]) / splitter.get_n_splits())
                    
                    for j in range(start, end - 1):
                        for k in range(j + 1, end):
                            self.assertDictEqual(tried_params[j], tried_params[k])
                    final_params.append(tried_params[start])
                
                if multiplier == 2:
                    final_scores = previous_final_scores + final_scores
                    final_params = previous_final_params + final_params
                
                self.assertEqual(len(fake_values) * multiplier, results.get_num_iterations())
                self.assertEqual(min(final_scores), results.get_best_score())
                best_config = results.get_best_config()
                del best_config["Duration"]
                del best_config["Score"]
                self.assertEqual(final_params[final_scores.index(min(final_scores))], best_config)
                
                previous_results = results
                previous_final_scores = final_scores
                previous_final_params = final_params
