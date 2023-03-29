import itertools
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from anomalearn.algorithms.postprocessing import BuilderVectorsSlidingWindow
from anomalearn.algorithms.preprocessing import SlidingWindowForecast, SlidingWindowReconstruct


class TestIntegrationBuilderVectorsSlidingWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(1000, 1)
        cls.series_multi = np.random.rand(1000, 6)

    def test_get_hyperparameters(self):
        shape_changer = SlidingWindowForecast(window=10)
        error_vectors1 = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
        hyper = error_vectors1.get_hyperparameters()

        self.assertIsInstance(hyper, dict)
        self.assertEqual(0, len(hyper))

    def test_set_hyperparameters(self):
        shape_changer = SlidingWindowForecast(window=10)
        error_vectors1 = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
        error_vectors1.set_hyperparameters({"false_att": "Ezio Auditore"})

        self.assertIs(shape_changer, error_vectors1.sliding_window)

    def test_equal(self):
        shape_changer = SlidingWindowForecast(window=10)
        shape_changer2 = SlidingWindowForecast(window=10)
        error_vectors1 = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
        error_vectors2 = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
        error_vectors3 = BuilderVectorsSlidingWindow(sliding_window=shape_changer2)
        
        self.assertEqual(error_vectors1, error_vectors2)
        self.assertEqual(error_vectors1, error_vectors3)
    
    def test_copy(self):
        shape_changer = SlidingWindowForecast(window=10)
        error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
        new = error_vectors.copy()
        
        self.assertEqual(error_vectors, new)
        self.assertIsNot(new, error_vectors)
    
    def test_save_and_load(self):
        shape_changer1 = SlidingWindowForecast(window=10)
        shape_changer2 = SlidingWindowForecast(window=30)
        error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer1)
        
        self.assertIs(error_vectors._sliding_window, shape_changer1)
        
        with TemporaryDirectory() as tmp_dir:
            error_vectors.save(tmp_dir)
            
            error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer2).load(tmp_dir)
        
            # When loading from file, it is impossible to keep track of the
            # reference between objects. However, the object should be created
            # such that it has a reference to an object equal to that at the time
            # of saving
            self.assertEqual(error_vectors._sliding_window, shape_changer1)
            
        shape_changer1 = SlidingWindowReconstruct(window=10)
        shape_changer2 = SlidingWindowReconstruct(window=30)
        error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer1)
        
        self.assertIs(error_vectors._sliding_window, shape_changer1)
        
        with TemporaryDirectory() as tmp_dir:
            error_vectors.save(tmp_dir)
            
            error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer2).load(tmp_dir)
        
            # When loading from file, it is impossible to keep track of the
            # reference between objects. However, the object should be created
            # such that it has a reference to an object equal to that at the time
            # of saving
            self.assertEqual(error_vectors._sliding_window, shape_changer1)
            
    def test_shape_change(self):
        for window, stride, forecast, series in itertools.product([10], [1, 2], [1, 2], [self.series_uni, self.series_multi]):
            print(f"test error vectors for forecasting building with window={window}, stride={stride}, forecast={forecast} and series={series.shape}")
            shape_changer = SlidingWindowForecast(window=window,
                                                  stride=stride,
                                                  forecast=forecast)
            error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
            
            new_x, new_y = shape_changer.shape_change(series)
            new_y_hat = new_y * -1
            vectors_y_hat, vectors_y = error_vectors.shape_change(new_y_hat, new_y)
            
            self.assertEqual(2, vectors_y_hat.ndim)
            self.assertEqual(series.shape[0], vectors_y_hat.shape[0])
            self.assertEqual(forecast * series.shape[1], vectors_y_hat.shape[1])
            self.assertTupleEqual(vectors_y_hat.shape, vectors_y.shape)
            
            not_skipped = (series.shape[0] - window - forecast) // stride + 1
            skipped = series.shape[0] - window - not_skipped - (forecast - 1)
            always_skipped = window * forecast
            self.assertEqual((forecast * (forecast - 1) + skipped * forecast + always_skipped) * new_y.shape[2], np.sum(np.isnan(vectors_y_hat)))
            self.assertEqual((forecast * (forecast - 1) + skipped * forecast + always_skipped) * new_y.shape[2], np.sum(np.isnan(vectors_y)))
            
            np.testing.assert_array_equal(vectors_y * -1, vectors_y_hat)
            
        for window, stride, series in itertools.product([10], [1, 2], [self.series_uni, self.series_multi]):
            print(f"test error vectors for reconstruction building with window={window}, stride={stride} and series={series.shape}")
            shape_changer = SlidingWindowReconstruct(window=window,
                                                     stride=stride)
            error_vectors = BuilderVectorsSlidingWindow(sliding_window=shape_changer)
            
            new_x, new_y = shape_changer.shape_change(series)
            new_y_hat = new_y * -1
            vectors_y_hat, vectors_y = error_vectors.shape_change(new_y_hat, new_y)
            
            self.assertEqual(2, vectors_y_hat.ndim)
            self.assertEqual(series.shape[0], vectors_y_hat.shape[0])
            self.assertEqual(window * series.shape[1], vectors_y_hat.shape[1])
            self.assertTupleEqual(vectors_y_hat.shape, vectors_y.shape)

            not_skipped = (series.shape[0] - window) // stride + 1
            skipped = series.shape[0] - not_skipped - (window - 1)
            self.assertEqual((window * (window - 1) + skipped * window) * new_y.shape[2], np.sum(np.isnan(vectors_y_hat)))
            self.assertEqual((window * (window - 1) + skipped * window) * new_y.shape[2], np.sum(np.isnan(vectors_y)))

            np.testing.assert_array_equal(vectors_y * -1, vectors_y_hat)
