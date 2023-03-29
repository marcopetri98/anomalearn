import itertools
import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from anomalearn.algorithms.models.time_series.anomaly.deep_learning.TSANeuralNetwork import StatesResetAtSpecifiedBatches
from tests.anomalearn.algorithms.models.time_series.anomaly.deep_learning.stubs import TSANeuralNetworkChild


def test_fitting_func(model: tf.keras.Model,
                      x_train: np.ndarray,
                      y_train: np.ndarray,
                      x_val: np.ndarray,
                      y_val: np.ndarray,
                      batch_size: int,
                      callbacks: list):
    TestTSANeuralNetwork.fitting_function_calls += 1
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mae")
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=1,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks,
                        verbose=False)
    return history


class TestTSANeuralNetwork(unittest.TestCase):
    fitting_function_calls = 0

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
        self.simpleModel.add(tf.keras.layers.Input((3, 10)))
        self.simpleModel.add(tf.keras.layers.Conv1D(10, 1))

    def test_save_and_load(self):
        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       test_fitting_func,
                                       window=10,
                                       prediction_horizon=3)
        tsa_nn.fit(self.rising_series, verbose=False)

        with TemporaryDirectory() as tmp_dir:
            tsa_nn.save(tmp_dir)

            contents = os.listdir(tmp_dir)
            self.assertNotEqual(0, len(contents))
            self.assertEqual(5, len(contents))
            self.assertIn(tsa_nn._TSANeuralNetwork__history_file, contents)
            self.assertIn(tsa_nn._TSANeuralNetwork__training_model_dir, contents)
            self.assertIn(tsa_nn._TSANeuralNetwork__trained_model_dir, contents)

            new_tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                               self.simpleModel,
                                               test_fitting_func,
                                               window=10,
                                               prediction_horizon=3)
            new_tsa_nn.load(tmp_dir)

            self.assertIsNone(new_tsa_nn.fitting_function)
            self.assertEqual(tsa_nn.prediction_horizon, new_tsa_nn.prediction_horizon)
            self.assertEqual(tsa_nn.validation_split, new_tsa_nn.validation_split)
            self.assertEqual(tsa_nn.mean_cov_sets, new_tsa_nn.mean_cov_sets)
            self.assertEqual(tsa_nn.threshold_sets, new_tsa_nn.threshold_sets)
            self.assertEqual(tsa_nn.window, new_tsa_nn.window)
            self.assertEqual(tsa_nn.stride, new_tsa_nn.stride)
            self.assertEqual(tsa_nn.batch_size, new_tsa_nn.batch_size)
            self.assertEqual(tsa_nn.stateful_model, new_tsa_nn.stateful_model)

            self.assertDictEqual(tsa_nn._fit_history, new_tsa_nn._fit_history)
            self.assertEqual(tsa_nn._initial_training_model.to_json(), new_tsa_nn._initial_training_model.to_json())
            self.assertEqual(tsa_nn._training_model.to_json(), new_tsa_nn._training_model.to_json())
            self.assertEqual(tsa_nn._prediction_model.to_json(), new_tsa_nn._prediction_model.to_json())

    def test_call_compute_mean_cov(self):
        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       window=10,
                                       prediction_horizon=3)

        samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, verbose=False)

        tsa_nn._call_compute_mean_cov(samples, targets, self.rising_series.shape[0], verbose=False)
        self.assertIsNotNone(tsa_nn._mean)
        self.assertIsNotNone(tsa_nn._cov)

        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       window=10,
                                       prediction_horizon=3)

        samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, verbose=False)

        n = 3
        tsa_nn._call_compute_mean_cov([samples] * n, [targets] * n, [self.rising_series.shape[0]] * n, verbose=False)
        self.assertIsNotNone(tsa_nn._mean)
        self.assertIsNotNone(tsa_nn._cov)

    def test_call_learn_threshold(self):
        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       window=10,
                                       prediction_horizon=3)

        samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, verbose=False)

        tsa_nn._call_compute_mean_cov(samples, targets, self.rising_series.shape[0], verbose=False)
        tsa_nn._call_learn_threshold(samples, targets, self.rising_series.shape[0], verbose=False)
        self.assertIsNotNone(tsa_nn._threshold)

        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       window=10,
                                       prediction_horizon=3)

        samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, verbose=False)

        n = 3
        tsa_nn._call_compute_mean_cov([samples] * n, [targets] * n, [self.rising_series.shape[0]] * n, verbose=False)
        tsa_nn._call_learn_threshold([samples] * n, [targets] * n, [self.rising_series.shape[0]] * n, verbose=False)
        self.assertIsNotNone(tsa_nn._threshold)

    def test_anomaly_score(self):
        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       window=10,
                                       prediction_horizon=3)
        samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, verbose=False)
        tsa_nn._call_compute_mean_cov(samples, targets, self.rising_series.shape[0], verbose=False)
        tsa_nn._call_learn_threshold(samples, targets, self.rising_series.shape[0], verbose=False)

        scores = tsa_nn.anomaly_score(self.rising_series, verbose=False)
        self.assertEqual(1, scores.ndim)
        self.assertEqual(self.rising_series.shape[0], scores.shape[0])

    def test_classify(self):
        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       window=10,
                                       prediction_horizon=3)
        samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, verbose=False)
        tsa_nn._call_compute_mean_cov(samples, targets, self.rising_series.shape[0], verbose=False)
        tsa_nn._call_learn_threshold(samples, targets, self.rising_series.shape[0], verbose=False)
        tsa_nn._threshold = round(self.rising_series.shape[0] / 2)
        tsa_nn.fake_scores = True

        labels = tsa_nn.classify(self.rising_series, verbose=False)
        self.assertEqual(1, labels.ndim)
        self.assertEqual(self.rising_series.shape[0], labels.shape[0])
        self.assertEqual(2, np.unique(labels).shape[0])
        self.assertIn(0, np.unique(labels).tolist())
        self.assertIn(1, np.unique(labels).tolist())

    def test_fit_and_complete(self):
        # only the training set is used to estimate the threshold and/or validation
        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       test_fitting_func,
                                       window=10,
                                       prediction_horizon=3)

        for cov_sets, threshold_sets in itertools.product(["training", "validation", "both"], ["training", "validation", "both"]):
            print(f"Checking fit_and_complete with mean_cov_sets={cov_sets}, and threshold_sets={threshold_sets}")
            tsa_nn.mean_cov_sets = cov_sets
            tsa_nn.threshold_sets = threshold_sets
            input_samples = 0

            samples, targets = tsa_nn._build_x_y_sequences(self.rising_series, False, False)
            match cov_sets:
                case "training":
                    input_samples += samples[:-10].shape[0]

                case "validation":
                    input_samples += samples[-10:].shape[0]

                case "both":
                    input_samples += samples.shape[0]

            match threshold_sets:
                case "training":
                    input_samples += samples[:-10].shape[0]

                case "validation":
                    input_samples += samples[-10:].shape[0]

                case "both":
                    input_samples += samples.shape[0]

            tsa_nn.reset_flags_counters()
            self.fitting_function_calls = 0
            tsa_nn._fit_and_complete(samples[:-10],
                                     targets[:-10],
                                     samples[-10:],
                                     targets[-10:],
                                     [],
                                     self.rising_series.shape[0],
                                     verbose=False)
            self.assertEqual(2 * self.rising_series.shape[0], tsa_nn.num_of_points_called)
            self.assertEqual(input_samples, tsa_nn.total_samples_input)

            tsa_nn.reset_flags_counters()
            self.fitting_function_calls = 0
            list_len = 3
            tsa_nn._fit_and_complete(samples[:-10],
                                     targets[:-10],
                                     samples[-10:],
                                     targets[-10:],
                                     [],
                                     [self.rising_series.shape[0]] * list_len,
                                     samples_train_list=[samples[:-10]] * list_len,
                                     targets_train_list=[targets[:-10]] * list_len,
                                     samples_valid_list=[samples[-10:]] * list_len,
                                     targets_valid_list=[targets[-10:]] * list_len,
                                     verbose=False)
            self.assertEqual(list_len * 2 * self.rising_series.shape[0], tsa_nn.num_of_points_called)
            self.assertEqual(list_len * input_samples, tsa_nn.total_samples_input)

    def test_fit(self):
        big_series = np.random.rand(10000, self.rising_series.shape[1])

        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       batch_size=2,
                                       window=10,
                                       prediction_horizon=3)

        for i, batch_size in enumerate([2, 1, 3, 7, 11, 13, 17, 21, 29]):
            print(f"Check that a stateful model will receive correct shapes in input with stateful={i!=0} and batch_size={batch_size}")
            tsa_nn.reset_flags_counters()
            tsa_nn.batch_size = batch_size
            tsa_nn.skip_fit_and_complete = True
            # set the stateful model and verify that the batch size is kept after first iteration to share lines of code
            tsa_nn.stateful_model = i != 0
            tsa_nn.fit(big_series, verbose=False)
            self.assertIsNotNone(tsa_nn.fit_and_complete_args)
            self.assertEqual(0, len(tsa_nn.fit_and_complete_kwargs))

            samples_train, targets_train, samples_valid, targets_valid, callbacks, num_of_points, verbose = tsa_nn.fit_and_complete_args
            self.assertEqual(big_series.shape[0], num_of_points)

            # assert dimensions and number of points
            self.assertEqual(3, samples_train.ndim)
            self.assertEqual(3, targets_train.ndim)
            self.assertEqual(3, samples_valid.ndim)
            self.assertEqual(3, targets_valid.ndim)
            self.assertEqual(samples_train.shape[0], targets_train.shape[0])
            self.assertEqual(samples_valid.shape[0], targets_valid.shape[0])

            # assert features, input and output
            self.assertEqual(samples_train.shape[1], samples_valid.shape[1])
            self.assertEqual(samples_train.shape[2], samples_valid.shape[2])
            self.assertEqual(targets_train.shape[1], targets_valid.shape[1])
            self.assertEqual(targets_train.shape[2], targets_valid.shape[2])

            if tsa_nn.stateful_model:
                # check the correct shape of the input
                self.assertEqual(0, samples_train.shape[0] % tsa_nn.batch_size)
                self.assertEqual(0, samples_valid.shape[0] % tsa_nn.batch_size)

    def test_fit_multiple(self):
        big_series = np.random.rand(200, self.rising_series.shape[1])

        tsa_nn = TSANeuralNetworkChild(self.simpleModel,
                                       self.simpleModel,
                                       self.empty_callable,
                                       batch_size=2,
                                       window=10,
                                       prediction_horizon=3)

        list_len = 3

        for i, batch_size in enumerate([2, 1, 2, 3, 5, 7, 9, 11, 13, 17, 19, 29]):
            print(f"Test fit_multiple with stateful={i!=0} and batch_size={batch_size}")
            tsa_nn.reset_flags_counters()
            tsa_nn.skip_fit_and_complete = True
            tsa_nn.batch_size = batch_size
            # set the stateful model and verify that the batch size is kept after first iteration to share lines of code
            tsa_nn.stateful_model = i != 0
            tsa_nn.fit_multiple([big_series] * list_len, verbose=False)
            self.assertIsNotNone(tsa_nn.fit_and_complete_args)
            self.assertEqual(4, len(tsa_nn.fit_and_complete_kwargs))

            samples_train, targets_train, samples_valid, targets_valid, callbacks, num_of_points, verbose = tsa_nn.fit_and_complete_args
            samples_train_list = tsa_nn.fit_and_complete_kwargs["samples_train_list"]
            targets_train_list = tsa_nn.fit_and_complete_kwargs["targets_train_list"]
            samples_valid_list = tsa_nn.fit_and_complete_kwargs["samples_valid_list"]
            targets_valid_list = tsa_nn.fit_and_complete_kwargs["targets_valid_list"]
            self.assertTrue(isinstance(num_of_points, list))
            self.assertEqual(list_len, len(num_of_points))
            self.assertListEqual([big_series.shape[0]] * list_len, num_of_points)
            self.assertEqual(1, len(callbacks))
            self.assertTrue(isinstance(callbacks[0], StatesResetAtSpecifiedBatches))
            self.assertTrue(isinstance(samples_train, np.ndarray))
            self.assertTrue(isinstance(targets_train, np.ndarray))
            self.assertTrue(isinstance(samples_valid, np.ndarray))
            self.assertTrue(isinstance(targets_valid, np.ndarray))
            self.assertTrue(isinstance(samples_train_list, list))
            self.assertTrue(isinstance(targets_train_list, list))
            self.assertTrue(isinstance(samples_valid_list, list))
            self.assertTrue(isinstance(targets_valid_list, list))

            self.assertEqual(samples_train_list[0].shape[1], samples_train.shape[1])
            self.assertEqual(samples_train_list[0].shape[2], samples_train.shape[2])
            self.assertEqual(targets_train_list[0].shape[1], targets_train.shape[1])
            self.assertEqual(targets_train_list[0].shape[2], targets_train.shape[2])
            self.assertEqual(samples_valid_list[0].shape[1], samples_valid.shape[1])
            self.assertEqual(samples_valid_list[0].shape[2], samples_valid.shape[2])
            self.assertEqual(targets_valid_list[0].shape[1], targets_valid.shape[1])
            self.assertEqual(targets_valid_list[0].shape[2], targets_valid.shape[2])

            self.assertEqual(sum([el.shape[0] for el in samples_train_list]), samples_train.shape[0])
            self.assertEqual(sum([el.shape[0] for el in targets_train_list]), targets_train.shape[0])
            self.assertEqual(sum([el.shape[0] for el in samples_valid_list]), samples_valid.shape[0])
            self.assertEqual(sum([el.shape[0] for el in targets_valid_list]), targets_valid.shape[0])

            self.assertEqual(3, samples_train.ndim)
            self.assertEqual(3, targets_train.ndim)
            self.assertEqual(3, samples_valid.ndim)
            self.assertEqual(3, targets_valid.ndim)
            self.assertEqual(samples_train.shape[0], targets_train.shape[0])
            self.assertEqual(samples_valid.shape[0], targets_valid.shape[0])

            self.assertEqual(samples_train.shape[1], samples_valid.shape[1])
            self.assertEqual(samples_train.shape[2], samples_valid.shape[2])
            self.assertEqual(targets_train.shape[1], targets_valid.shape[1])
            self.assertEqual(targets_train.shape[2], targets_valid.shape[2])

            if tsa_nn.stateful_model:
                self.assertEqual(0, samples_train.shape[0] % tsa_nn.batch_size)
                self.assertEqual(0, samples_valid.shape[0] % tsa_nn.batch_size)
                self.assertEqual(0, targets_train.shape[0] % tsa_nn.batch_size)
                self.assertEqual(0, targets_valid.shape[0] % tsa_nn.batch_size)

                for x_tr, y_tr, x_va, y_va in zip(samples_train_list, targets_train_list, samples_valid_list, targets_valid_list):
                    self.assertEqual(0, x_tr.shape[0] % tsa_nn.batch_size)
                    self.assertEqual(0, y_tr.shape[0] % tsa_nn.batch_size)
                    self.assertEqual(0, x_va.shape[0] % tsa_nn.batch_size)
                    self.assertEqual(0, y_va.shape[0] % tsa_nn.batch_size)
