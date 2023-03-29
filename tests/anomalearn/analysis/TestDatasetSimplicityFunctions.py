import time
import unittest
from pathlib import Path

import numpy as np

from anomalearn.analysis import analyse_constant_simplicity, \
    analyse_mov_avg_simplicity, analyse_mov_std_simplicity, \
    analyse_mixed_simplicity
from anomalearn.reader.time_series import SMDReader
from anomalearn.utils import load_py_json


def key_order(x):
    try:
        return int(x.name.split(".")[0].split("_")[-1])
    except ValueError:
        return int(x.name.split(".")[0].split("_")[-2])


class TestDatasetSimplicityFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.uni_series = np.arange(100).reshape(-1, 1)
        self.uni_labels = np.zeros(self.uni_series.shape[0])
        self.multi_series = np.array([np.arange(100), np.arange(100)]).transpose()
        self.multi_labels = np.zeros(self.multi_series.shape[0])
        
        # the resolution at which two bounds are considered equal
        self.resolution = 1e-15
        
    def assert_constant_results(self, results, exp_results):
        self.assertSetEqual({"constant_score", "upper_bound", "lower_bound", "diff_order"}, set(results.keys()))
        self.assertEqual(exp_results["constant_score"], results["constant_score"])
        self.assertEqual(exp_results["diff_order"], results["diff_order"])
        self.assertListEqual(exp_results["upper_bound"], results["upper_bound"])
        self.assertListEqual(exp_results["lower_bound"], results["lower_bound"])
        
    def assert_movement_results(self, results, exp_results, score_name):
        self.assertSetEqual({score_name, "upper_bound", "lower_bound", "diff_order", "window"}, set(results.keys()))
        self.assertEqual(exp_results[score_name], results[score_name])
        self.assertEqual(exp_results["diff_order"], results["diff_order"])
        self.assertEqual(exp_results["window"], results["window"])
        for i, (el1, el2) in enumerate(zip(exp_results["lower_bound"], results["lower_bound"])):
            if (el1 is None) != (el2 is None):
                raise AssertionError(f"Left at index {i} ({el1}) is not equal to right at index {i} ({el2})")
            elif el1 is not None:
                self.assertGreaterEqual(self.resolution, el1 - el2)
        for i, (el1, el2) in enumerate(zip(exp_results["upper_bound"], results["upper_bound"])):
            if (el1 is None) != (el2 is None):
                raise AssertionError(f"Left at index {i} ({el1}) is not equal to right at index {i} ({el2})")
            elif el1 is not None:
                self.assertGreaterEqual(self.resolution, el1 - el2)
        
    def test_analyse_constant_simplicity(self):
        # the cases loaded from file should be such that:
        # case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None)
        # case 1: score 1, diff 0, lower bound
        # case 2: score 1, diff 0, upper bound
        # case 3: score 1, diff 0, both bounds
        # case 4: score 1, diff 1, lower bound
        # case 5: score 1, diff 1, upper bound
        # case 6: score 1, diff 1, both bounds
        # case 7: 0 < score < 1, diff 0, lower bound
        # case 8: 0 < score < 1, diff 0, upper bound
        # case 9: 0 < score < 1, diff 0, both bounds
        # case 10: 0 < score < 1, diff 1, lower bound
        # case 11: 0 < score < 1, diff 1, upper bound
        # case 12: 0 < score < 1, diff 1, both bounds
        test_data = Path(__file__).parent / "test_data" / "constant_simplicity"
        cases = sorted([e for e in test_data.glob("const_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("const_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("const_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...", end="\n\n")
            results = analyse_constant_simplicity(series, labels, diff=3)
            self.assert_constant_results(results, exp_results)
    
    def test_analyse_mov_avg_simplicity(self):
        # the cases loaded from file should be such that:
        # case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
        # case 1: score 1, diff 0, lower bound, window 2
        # case 2: score 1, diff 0, upper bound, window 2
        # case 3: score 1, diff 0, both bounds, window 2
        # case 4: score 1, diff 1, lower bound, window 2
        # case 5: score 1, diff 1, upper bound, window 2
        # case 6: score 1, diff 1, both bounds, window 2
        # case 7: 0 < score < 1, diff 0, lower bound, window 2
        # case 8: 0 < score < 1, diff 0, upper bound, window 2
        # case 9: 0 < score < 1, diff 0, both bounds, window 2
        # case 10: 0 < score < 1, diff 1, lower bound, window 2
        # case 11: 0 < score < 1, diff 1, upper bound, window 2
        # case 12: 0 < score < 1, diff 1, both bounds, window 2
        # case 13: score 1, diff 0, lower bound, window >2
        # case 14: score 1, diff 0, upper bound, window >2
        # case 15: score 1, diff 0, both bounds, window >2
        # case 16: score 1, diff 1, lower bound, window >2
        # case 17: score 1, diff 1, upper bound, window >2
        # case 18: score 1, diff 1, both bounds, window >2
        # case 19: 0 < score < 1, diff 0, lower bound, window >2
        # case 20: 0 < score < 1, diff 0, upper bound, window >2
        # case 21: 0 < score < 1, diff 0, both bounds, window >2
        # case 22: 0 < score < 1, diff 1, lower bound, window >2
        # case 23: 0 < score < 1, diff 1, upper bound, window >2
        # case 24: 0 < score < 1, diff 1, both bounds, window >2
        test_data = Path(__file__).parent / "test_data" / "mov_avg_simplicity"
        cases = sorted([e for e in test_data.glob("mov_avg_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("mov_avg_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("mov_avg_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...", end="\n\n")
            results = analyse_mov_avg_simplicity(series, labels, window_range=(2, 100), diff=3)
            self.assert_movement_results(results, exp_results, "mov_avg_score")
    
    def test_analyse_mov_std_simplicity(self):
        # the cases loaded from file should be such that:
        # case 0: score 0 (therefore diff=0, lower_bound=None, upper_bound=None, window=2)
        # case 1: score 1, diff 0, lower bound, window 2
        # case 2: score 1, diff 0, upper bound, window 2
        # case 3: score 1, diff 0, both bounds, window 2
        # case 4: score 1, diff 1, lower bound, window 2
        # case 5: score 1, diff 1, upper bound, window 2
        # case 6: score 1, diff 1, both bounds, window 2
        # case 7: 0 < score < 1, diff 0, lower bound, window 2
        # case 8: 0 < score < 1, diff 0, upper bound, window 2
        # case 9: 0 < score < 1, diff 0, both bounds, window 2
        # case 10: 0 < score < 1, diff 1, lower bound, window 2
        # case 11: 0 < score < 1, diff 1, upper bound, window 2
        # case 12: 0 < score < 1, diff 1, both bounds, window 2
        # case 13: score 1, diff 0, lower bound, window >2
        # case 14: score 1, diff 0, upper bound, window >2
        # case 15: score 1, diff 0, both bounds, window >2
        # case 16: score 1, diff 1, lower bound, window >2
        # case 17: score 1, diff 1, upper bound, window >2
        # case 18: score 1, diff 1, both bounds, window >2
        # case 19: 0 < score < 1, diff 0, lower bound, window >2
        # case 20: 0 < score < 1, diff 0, upper bound, window >2
        # case 21: 0 < score < 1, diff 0, both bounds, window >2
        # case 22: 0 < score < 1, diff 1, lower bound, window >2
        # case 23: 0 < score < 1, diff 1, upper bound, window >2
        # case 24: 0 < score < 1, diff 1, both bounds, window >2
        test_data = Path(__file__).parent / "test_data" / "mov_std_simplicity"
        cases = sorted([e for e in test_data.glob("mov_std_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("mov_std_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("mov_std_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...", end="\n\n")
            results = analyse_mov_std_simplicity(series, labels, window_range=(2, 100), diff=3)
            self.assert_movement_results(results, exp_results, "mov_std_score")

    def test_mixed_simplicity(self):
        # tests that all cases of constant, moving average and moving standard
        # deviation cases are correctly computed
        test_data = Path(__file__).parent / "test_data" / "constant_simplicity"
        cases = sorted([e for e in test_data.glob("const_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("const_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("const_case_*[0-9]_result.json")], key=key_order)
        
        test_data = Path(__file__).parent / "test_data" / "mov_avg_simplicity"
        cases += sorted([e for e in test_data.glob("mov_avg_case_*[0-9].csv")], key=key_order)
        cases_labels += sorted([e for e in test_data.glob("mov_avg_case_*[0-9]_labels.csv")], key=key_order)
        cases_results += sorted([e for e in test_data.glob("mov_avg_case_*[0-9]_result.json")], key=key_order)
        
        test_data = Path(__file__).parent / "test_data" / "mov_std_simplicity"
        cases += sorted([e for e in test_data.glob("mov_std_case_*[0-9].csv")], key=key_order)
        cases_labels += sorted([e for e in test_data.glob("mov_std_case_*[0-9]_labels.csv")], key=key_order)
        cases_results += sorted([e for e in test_data.glob("mov_std_case_*[0-9]_result.json")], key=key_order)
        
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
            
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...")
            results = analyse_mixed_simplicity(series, labels, window_range=(2, 100), diff=3)
            self.assertSetEqual({"mixed_score", "const_result", "mov_avg_result", "mov_std_result"}, set(results.keys()))
            if str(case.name).startswith("const"):
                print(f"Asserting the constant results...")
                self.assert_constant_results(results["const_result"], exp_results)
                self.assertGreaterEqual(results["mixed_score"], results["const_result"]["constant_score"])
            if str(case.name).startswith("mov_avg"):
                print(f"Asserting the moving average results...")
                self.assert_movement_results(results["mov_avg_result"], exp_results, "mov_avg_score")
                self.assertGreaterEqual(results["mixed_score"], results["mov_avg_result"]["mov_avg_score"])
            if str(case.name).startswith("mov_std"):
                print(f"Asserting the moving standard deviation results...")
                self.assert_movement_results(results["mov_std_result"], exp_results, "mov_std_score")
                self.assertGreaterEqual(results["mixed_score"], results["mov_std_result"]["mov_std_score"])
                
            # check that 0 score cases are 0 for mixed score if all are 0,
            # that is cases 0 and their extension to multivariate
            if results["const_result"]["constant_score"] == 0 and results["mov_avg_result"]["mov_avg_score"] == 0 and results["mov_std_result"]["mov_std_score"] == 0:
                print(f"Asserting that the mixed score is 0...")
                self.assertEqual(0, results["mixed_score"])
                
            # check that 1 score cases are 1 for mixed score if all at least
            # one is 1, that is cases with expected score 1
            if results["const_result"]["constant_score"] == 1 or results["mov_avg_result"]["mov_avg_score"] == 1 or results["mov_std_result"]["mov_std_score"] == 1:
                print(f"Asserting that the mixed score is 1...")
                self.assertEqual(1, results["mixed_score"])
                
            print("\n")

        # check that in a mixed case in which single scores fall in (0,1) and
        # no one is either 0 or 1 the mixed score is correct
        test_data = Path(__file__).parent / "test_data" / "mixed_simplicity"
        cases = sorted([e for e in test_data.glob("mixed_case_*[0-9].csv")], key=key_order)
        cases_labels = sorted([e for e in test_data.glob("mixed_case_*[0-9]_labels.csv")], key=key_order)
        cases_results = sorted([e for e in test_data.glob("mixed_case_*[0-9]_result.json")], key=key_order)
        for case, label, result in zip(cases, cases_labels, cases_results):
            print(f"Reading {case.name}, {label.name}, {result.name}")
            series = np.genfromtxt(case, delimiter=",")
            labels = np.genfromtxt(label, delimiter=",")
            exp_results = load_py_json(result)
    
            if series.ndim == 1:
                series = series.reshape((-1, 1))

            print(f"Asserting the results...")
            results = analyse_mixed_simplicity(series, labels, window_range=(2, 100), diff=3)
            self.assertEqual(exp_results["mixed_score"], results["mixed_score"])

            print("\n")

    def test_speed(self):
        standard_path = Path(__file__).parent / "../../../data/anomaly_detection/smd"
        reader = SMDReader(str(standard_path))
        big_series = reader.read("machine-1-1", verbose=False).get_dataframe()
        values = big_series[sorted(set(big_series.columns).difference(["class", "timestamp", "is_training", "interpretation"]),
                                   key=lambda x: int(x.split("_")[-1]))].values
        labels = big_series["class"].values
        values = np.ascontiguousarray(values, dtype=values.dtype)
        labels = np.ascontiguousarray(labels, dtype=labels.dtype)

        # compile the functions before testing speed
        print("Compiling functions before testing the speed.", end="\n\n")
        dummy_series = np.random.rand(100, 1)
        dummy_labels = np.zeros(100)
        dummy_labels[50] = 1
        _ = analyse_constant_simplicity(dummy_series, dummy_labels)
        _ = analyse_mov_avg_simplicity(dummy_series, dummy_labels)
        _ = analyse_mov_std_simplicity(dummy_series, dummy_labels)
        _ = analyse_mixed_simplicity(dummy_series, dummy_labels)

        print(f"Start to analyse constant simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_constant_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")

        print(f"Start to analyse moving average simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_mov_avg_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")

        print(f"Start to analyse moving standard deviation simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_mov_std_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")

        print(f"Start to analyse mixed simplicity of series of shape {values.shape}")
        start_time = time.time()
        results = analyse_mixed_simplicity(values, labels)
        end_time = time.time()
        print(f"\tTime elapsed: {end_time - start_time}.")
        print(f"\tResults: {results}", end="\n\n")
