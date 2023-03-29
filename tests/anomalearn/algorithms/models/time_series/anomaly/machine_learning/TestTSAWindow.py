import math
import os.path
import pathlib
import unittest

import numpy as np
import pandas as pd

from tests.anomalearn.algorithms.models.time_series.anomaly.machine_learning.stubs import TSAWindowChild


class TestTSAWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        square_wave_path = "test_data/square_wave.csv"
        dir_path = pathlib.Path(__file__).parent.resolve()
        
        cls.square_wave_path = os.path.join(dir_path, square_wave_path)
        cls.square_wave_df = pd.read_csv(cls.square_wave_path)
        
    def setUp(self) -> None:
        self.time_series = self.square_wave_df["value"].copy()
        self.time_series_len = self.time_series.values.shape[0]
        self.targets = self.square_wave_df["target"].copy()
        
    def test_space_projection(self):
        configurations = [(2, 1), (3, 1), (4, 1), (5, 1),
                          (2, 2), (4, 2), (6, 2), (7, 3), (7, 5), (8, 2),
                          (2, 2), (11, 11),
                          (2, 3), (2, 4), (2, 5), (2, 10)]

        tsa_window_stub = TSAWindowChild(window=(self.time_series_len + 1))
        with self.assertRaises(ValueError):
            vector_data, num_windows = tsa_window_stub._project_time_series(self.time_series.values.reshape(-1, 1))

        tsa_window_stub = TSAWindowChild(window=3)
        with self.assertRaises(ValueError):
            vector_data, num_windows = tsa_window_stub._project_time_series(self.time_series.values.reshape(-1, 2))
        
        for window, stride in configurations:
            tsa_window_stub = TSAWindowChild(window=window, stride=stride)
            vector_data, num_windows = tsa_window_stub._project_time_series(self.time_series.values.reshape(-1, 1))
            
            self.assertEqual(self.time_series_len, num_windows.shape[0])
            self.assertEqual(math.floor((self.time_series_len - window) / stride) + 1, vector_data.shape[0])
            self.assertEqual(window, vector_data.shape[1])
            
            # these formulas can be easily demonstrated mathematically
            if stride > window:
                for idx, num_eval in enumerate(num_windows):
                    start_w = math.floor(idx / stride)
                    if stride * start_w <= idx <= stride * start_w + window - 1 < self.time_series_len:
                        self.assertEqual(1, num_eval)
                    else:
                        self.assertEqual(0, num_eval)
            else:
                if window == stride:
                    self.assertEqual(np.sum(num_windows), self.time_series_len)
                elif window % stride == 0:
                    for idx, num_eval in enumerate(num_windows):
                        self.assertEqual(min(window / stride, math.floor(idx / stride) + 1, math.floor((self.time_series_len - 1 - idx) / stride) + 1), num_eval)
                else:
                    for idx, num_eval in enumerate(num_windows):
                        w_over_s = math.floor(window / stride)
                        if math.floor(idx / stride) + 1 <=  w_over_s:
                            self.assertEqual(math.floor(idx / stride) + 1, num_eval)
                        elif math.floor((self.time_series_len - 1 - idx) / stride) + 1 <= w_over_s:
                            self.assertEqual(math.floor((self.time_series_len - 1 - idx) / stride) + 1, num_eval)
                        elif idx < window and stride - (window - idx) < w_over_s:
                            self.assertEqual(w_over_s, num_eval)
                        elif idx < window and stride - (window - idx) >= w_over_s:
                            self.assertEqual(w_over_s + 1, num_eval)
                        elif stride - (idx - window) % stride > window % stride:
                            self.assertEqual(w_over_s, num_eval)
                        else:
                            self.assertEqual(w_over_s + 1, num_eval)
    
    def test_scoring_centre(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="centre")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([np.nan, 1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1, np.nan])

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_scoring_left(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="left")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1, np.nan, np.nan])

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_scoring_right(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="right")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([np.nan, np.nan, 1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1])

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_scoring_min(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="min")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([1, 1, 1, 2, 1, 1, 1, 3, 6, 3, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1])

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_scoring_max(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="max")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([1, 2, 3, 3, 3, 3, 6, 9, 9, 9, 6, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 1])

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_scoring_average(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="average")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([1, 1.5, 2, 7/3, 2, 2, 10/3, 6, 7, 6, 10/3, 2, 2, 7/3, 2, 4/3, 4/3, 2, 7/3, 2, 1.5, 1])

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_scoring_minmax(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="minmax",
                                         scoring="max")
        # window_scores = [1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1]
        expected_point_scores = np.array([1, 2, 3, 3, 3, 3, 6, 9, 9, 9, 6, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 1])
        expected_point_scores = expected_point_scores - 1
        expected_point_scores = expected_point_scores / (9 - 1)

        point_scores = tsa_window_stub.anomaly_score(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_scores == point_scores) | (np.isnan(expected_point_scores) & np.isnan(point_scores))).all())
    
    def test_labelling_centre(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         classification="centre")
        window_scores = np.array([1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1])
        window_labels = window_scores > np.mean(window_scores)
        expected_point_labels = np.array(window_labels, dtype=float)
        expected_point_labels = np.insert(expected_point_labels, 0, np.nan)
        expected_point_labels = np.append(expected_point_labels, np.nan)

        point_labels = tsa_window_stub.classify(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_labels == point_labels) | (np.isnan(expected_point_labels) & np.isnan(point_labels))).all())
    
    def test_labelling_left(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         classification="left")
        window_scores = np.array([1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1])
        window_labels = window_scores > np.mean(window_scores)
        expected_point_labels = np.array(window_labels, dtype=float)
        expected_point_labels = np.append(expected_point_labels, np.nan)
        expected_point_labels = np.append(expected_point_labels, np.nan)

        point_labels = tsa_window_stub.classify(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_labels == point_labels) | (np.isnan(expected_point_labels) & np.isnan(point_labels))).all())
    
    def test_labelling_right(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         classification="right")
        window_scores = np.array([1, 2, 3, 2, 1, 3, 6, 9, 6, 3, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1])
        window_labels = window_scores > np.mean(window_scores)
        expected_point_labels = np.array(window_labels, dtype=float)
        expected_point_labels = np.insert(expected_point_labels, 0, np.nan)
        expected_point_labels = np.insert(expected_point_labels, 0, np.nan)

        point_labels = tsa_window_stub.classify(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_point_labels == point_labels) | (np.isnan(expected_point_labels) & np.isnan(point_labels))).all())
    
    def test_labelling_point_score(self):
        threshold = 2.5
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         scaling="none",
                                         scoring="min",
                                         classification="points_score",
                                         threshold=threshold)
        expected_point_scores = np.array([1, 1, 1, 2, 1, 1, 1, 3, 6, 3, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1])
        expected_labels = expected_point_scores > threshold

        point_labels = tsa_window_stub.classify(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_labels == point_labels) | (np.isnan(expected_labels) & np.isnan(point_labels))).all())
    
    def test_labelling_majority_voting(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         classification="majority_voting")
        # window_labels = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        expected_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        point_labels = tsa_window_stub.classify(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_labels == point_labels) | (np.isnan(expected_labels) & np.isnan(point_labels))).all())
    
    def test_labelling_voting(self):
        tsa_window_stub = TSAWindowChild(window=3,
                                         stride=1,
                                         classification="voting",
                                         threshold=0.3)
        # window_labels = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        expected_labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0])

        point_labels = tsa_window_stub.classify(self.time_series.values.reshape(-1, 1))
        
        self.assertTrue(((expected_labels == point_labels) | (np.isnan(expected_labels) & np.isnan(point_labels))).all())
    
    def test_set_parameters(self):
        tsa_window_stub = TSAWindowChild()
        tsa_window_stub.set_params(window=10,
                                   stride=2,
                                   scaling="none",
                                   scoring="min",
                                   classification="majority_voting",
                                   threshold=0.5)
        
        self.assertEqual(10, tsa_window_stub.window)
        self.assertEqual(2, tsa_window_stub.stride)
        self.assertEqual("none", tsa_window_stub.scaling)
        self.assertEqual("min", tsa_window_stub.scoring)
        self.assertEqual("majority_voting", tsa_window_stub.classification)
        self.assertEqual(0.5, tsa_window_stub.threshold)
    
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(scoring="NOT PRESENT")
        
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(scaling="NOT PRESENT")

        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(classification="NOT PRESENT")
            
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(window=-10)
            
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(stride=-10)
            
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(classification="voting", threshold=55)
        
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(scoring="centre", window=2)
        
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(classification="centre", window=2)
        
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(scoring="centre", window=3, stride=2)
        
        with self.assertRaises(ValueError):
            tsa_window_stub = TSAWindowChild(classification="centre", window=3, stride=2)

        tsa_window_stub = TSAWindowChild(scoring="left", window=2)
        tsa_window_stub = TSAWindowChild(scoring="right", window=2)
        tsa_window_stub = TSAWindowChild(scoring="min", window=2)
        tsa_window_stub = TSAWindowChild(scoring="max", window=2)
        tsa_window_stub = TSAWindowChild(scoring="average", window=2)
        tsa_window_stub = TSAWindowChild(classification="left", window=2)
        tsa_window_stub = TSAWindowChild(classification="right", window=2)
