import itertools
import unittest

import numpy as np
import tensorflow as tf

from tests.anomalearn.algorithms.models.time_series.anomaly.deep_learning.stubs import TSANNStandardChild, FakeModel


class TestTSANNStandard(unittest.TestCase):
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

    def test_remove_extra_points(self):
        values = np.random.rand(1000, 5, 10)
        objectives = np.random.rand(1000, 5, 10)
        for batch_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 19, 23, 29, 41, 47, 59]:
            print(f"Testing remove extra points with batch_size={batch_size}")
            tsa_nn = TSANNStandardChild(self.simpleModel,
                                        self.simpleModel,
                                        self.empty_callable,
                                        batch_size=batch_size)

            samples, targets = tsa_nn._remove_extra_points(values, objectives, False)
            self.assertEqual(samples.shape[0], targets.shape[0])
            self.assertEqual(samples.shape[1], values.shape[1])
            self.assertEqual(samples.shape[2], values.shape[2])
            self.assertEqual(targets.shape[1], objectives.shape[1])
            self.assertEqual(targets.shape[2], objectives.shape[2])
            self.assertGreaterEqual(values.shape[0], samples.shape[0])
            self.assertGreaterEqual(objectives.shape[0], targets.shape[0])

            self.assertEqual((values.shape[0] // batch_size) * batch_size, samples.shape[0])
            self.assertEqual((objectives.shape[0] // batch_size) * batch_size, targets.shape[0])

    def test_get_pred_true_vectors(self):
        for window, stride in itertools.product([1, 5, 10, 20], [1, 2, 5, 10]):
            print(f"Testing get pred true vectors with window={window}, stride={stride}")
            tsa_nn = TSANNStandardChild(self.simpleModel,
                                        self.simpleModel,
                                        self.empty_callable,
                                        window=window,
                                        stride=stride)
            fake_model = FakeModel(((self.rising_series.shape[0] - window) // stride + 1, self.rising_series.shape[1], window))
            tsa_nn._initial_training_model = fake_model
            tsa_nn._training_model = fake_model
            tsa_nn._prediction_model = fake_model

            samples, targets = tsa_nn._build_x_y_sequences(self.rising_series)
            pred, targ = tsa_nn._get_pred_true_vectors(samples, targets, self.rising_series.shape[0])
            tsa_nn.reset_counters()
            self.assertEqual(1, fake_model.number_of_times_predict_called)
            self.assertEqual(0, tsa_nn.build_x_y_sequences_called)
            self.assertEqual(2, pred.ndim)
            self.assertEqual(2, targ.ndim)
            self.assertEqual((self.rising_series.shape[0], targets.shape[1] * window), pred.shape)
            self.assertEqual((self.rising_series.shape[0], targets.shape[1] * window), targ.shape)

            fake_model.reset_num_predict_called()
            lists = 3
            pred, targ = tsa_nn._get_pred_true_vectors([samples] * lists, [targets] * lists, [self.rising_series.shape[0]] * lists)
            self.assertEqual(lists, fake_model.number_of_times_predict_called)
            self.assertEqual(0, tsa_nn.build_x_y_sequences_called)
            self.assertEqual(2, pred.ndim)
            self.assertEqual(2, targ.ndim)
            self.assertEqual((self.rising_series.shape[0] * lists, targets.shape[1] * window), pred.shape)
            self.assertEqual((self.rising_series.shape[0] * lists, targets.shape[1] * window), targ.shape)

    def test_predict(self):
        for window, stride in itertools.product([1, 5, 10, 20], [1, 2, 5, 10]):
            print(f"Testing predict method with window={window}, stride={stride}")
            tsa_nn = TSANNStandardChild(self.simpleModel,
                                        self.simpleModel,
                                        self.empty_callable,
                                        window=window,
                                        stride=stride)
            fake_model = FakeModel(((self.rising_series.shape[0] - window) // stride + 1, self.rising_series.shape[1], window))
            tsa_nn._initial_training_model = fake_model
            tsa_nn._training_model = fake_model
            tsa_nn._prediction_model = fake_model

            pred = tsa_nn.predict(self.rising_series)
            self.assertEqual(self.rising_series.shape, pred.shape)
            self.assertEqual(1, tsa_nn.build_x_y_sequences_called)
            self.assertEqual(1, tsa_nn.get_pred_true_vectors_called)
