import itertools
import unittest

import numpy as np
import tensorflow as tf

from anomalearn.algorithms.models.time_series.anomaly.deep_learning import TSANNPredictor


class TestTSANNPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        points = 100
        features = 3
        
        rising_series = np.zeros((points, features))
        rising_series[:, 0] = np.arange(points)
        rising_series[:, 1] = np.arange(points)
        rising_series[:, 2] = np.arange(points)
        cls.rising_series = rising_series
    
    def setUp(self) -> None:
        self.empty_callable = lambda x: None
        self.simpleModel = tf.keras.Sequential()
    
    def test_build_x_y_sequences(self):
        # check that the building process works as desired
        for horizon, window, stride in itertools.product([1, 2, 5], [5, 10, 20], [1, 2, 5]):
            print(f"Testing construction of sequences with horizon={horizon}, window={window}, stride={stride}")
            
            model = TSANNPredictor(self.simpleModel,
                                   self.simpleModel,
                                   self.empty_callable,
                                   prediction_horizon=horizon,
                                   window=window,
                                   stride=stride)

            maximum_points = (self.rising_series.shape[0] - window - horizon) // stride + 1
            x, y = model._build_x_y_sequences(self.rising_series)
            self.assertEqual((maximum_points, self.rising_series.shape[1], window), x.shape)
            self.assertEqual((maximum_points, self.rising_series.shape[1], horizon), y.shape)
            
            for i, (x_line, y_line) in enumerate(zip(x, y)):
                start = i * stride
                np.testing.assert_array_equal(self.rising_series[start:start+window], np.transpose(x_line))
                np.testing.assert_array_equal(self.rising_series[start+window:start+window+horizon], np.transpose(y_line))

        # check that the function is capable of keeping the batches
        for horizon, window, stride, batch_size in itertools.product([1, 2, 5], [1, 5, 10, 20], [1, 2, 5], [1, 3, 4, 5, 7, 10, 13, 17, 21]):
            print(f"Testing construction of sequences keeping the batch size with horizon={horizon}, window={window}, stride={stride}, batch_size={batch_size}")

            model = TSANNPredictor(self.simpleModel,
                                   self.simpleModel,
                                   self.empty_callable,
                                   prediction_horizon=horizon,
                                   batch_size=batch_size,
                                   window=window,
                                   stride=stride)

            maximum_points = (self.rising_series.shape[0] - window - horizon) // stride + 1
            x, y = model._build_x_y_sequences(self.rising_series, keep_batches=True, verbose=False)
            self.assertEqual((maximum_points // batch_size) * batch_size, x.shape[0])
            self.assertEqual((maximum_points // batch_size) * batch_size, y.shape[0])
    
    def test_fill_pred_targ_matrices(self):
        for horizon, window, stride in itertools.product([1, 2, 5], [1, 5, 10, 20], [1, 2, 5]):
            print(f"Testing pred targ matrices filling with horizon={horizon}, window={window}, stride={stride}")

            model = TSANNPredictor(self.simpleModel,
                                   self.simpleModel,
                                   self.empty_callable,
                                   prediction_horizon=horizon,
                                   window=window,
                                   stride=stride)
            x, y = model._build_x_y_sequences(self.rising_series)
            pred = y * -1
            
            pred_mat = np.full((self.rising_series.shape[0], y.shape[1], y.shape[2]), np.nan)
            true_mat = np.full((self.rising_series.shape[0], y.shape[1], y.shape[2]), np.nan)
            
            model._fill_pred_targ_matrices(pred, y, pred_mat, true_mat)
            not_skipped = (self.rising_series.shape[0] - window - horizon) // stride + 1
            skipped = self.rising_series.shape[0] - window - not_skipped - (horizon - 1)
            always_skipped = window * horizon
            self.assertEqual((horizon * (horizon - 1) + skipped * horizon + always_skipped) * y.shape[1], np.sum(np.isnan(pred_mat)))
            self.assertEqual((horizon * (horizon - 1) + skipped * horizon + always_skipped) * y.shape[1], np.sum(np.isnan(true_mat)))
            np.testing.assert_array_equal(true_mat, pred_mat * -1)
