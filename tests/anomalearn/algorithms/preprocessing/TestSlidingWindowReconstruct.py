import itertools
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from skopt.space import Integer

from anomalearn.algorithms.preprocessing import SlidingWindowReconstruct


class TestSlidingWindowReconstruct(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(1000, 1)
        cls.series_multi = np.random.rand(1000, 6)

    def test_get_hyperparameters(self):
        shape_changer1 = SlidingWindowReconstruct(window=10, stride=1)
        hyper = shape_changer1.get_hyperparameters()

        self.assertEqual(2, len(hyper))
        for key, value in hyper.items():
            self.assertIsInstance(value, dict)
            self.assertEqual(2, len(value))
            self.assertIn("value", value)
            self.assertIn("set", value)

        self.assertEqual(10, hyper["window"]["value"])
        self.assertEqual(Integer(1, np.inf), hyper["window"]["set"])
        self.assertEqual(1, hyper["stride"]["value"])
        self.assertEqual(Integer(1, np.inf), hyper["stride"]["set"])

    def test_set_hyperparameters(self):
        shape_changer1 = SlidingWindowReconstruct(window=10, stride=1)
        shape_changer1.set_hyperparameters({"window": 5, "stride": 10})

        self.assertEqual(5, shape_changer1.window)
        self.assertEqual(10, shape_changer1.stride)
        
    def test_equality(self):
        shape_changer1 = SlidingWindowReconstruct(window=10, stride=1)
        shape_changer2 = SlidingWindowReconstruct(window=10, stride=1)
        
        shape_changer3 = SlidingWindowReconstruct(window=5, stride=1)
        shape_changer4 = SlidingWindowReconstruct(window=10, stride=5)
        shape_changer5 = SlidingWindowReconstruct(window=5, stride=5)
        
        self.assertEqual(shape_changer1, shape_changer2)
        self.assertNotEqual(shape_changer1, shape_changer3)
        self.assertNotEqual(shape_changer1, shape_changer4)
        self.assertNotEqual(shape_changer1, shape_changer5)
        
        self.assertNotEqual(shape_changer1, 1848)
        self.assertNotEqual(shape_changer1, None)
        self.assertNotEqual(shape_changer1, "Saruman The White")
        
    def test_copy(self):
        shape_changer = SlidingWindowReconstruct(window=15)
        new_obj = shape_changer.copy()
        
        self.assertEqual(shape_changer, new_obj)
        self.assertIsNot(shape_changer, new_obj)
    
    def test_save_and_load(self):
        shape_changer = SlidingWindowReconstruct(window=10, stride=2)
        
        self.assertIsNone(shape_changer.points_seen)
        self.assertEqual(10, shape_changer.window)
        self.assertEqual(2, shape_changer.stride)
        
        with TemporaryDirectory() as tmp_dir:
            shape_changer.save(tmp_dir)
            
            shape_changer = SlidingWindowReconstruct(window=-1, stride=-1)
            shape_changer._points_seen = "Gandalf The Grey"
            shape_changer.load(tmp_dir)
            
            self.assertIsNone(shape_changer.points_seen)
            self.assertEqual(10, shape_changer.window)
            self.assertEqual(2, shape_changer.stride)
            
        shape_changer = SlidingWindowReconstruct(window=1, stride=1)
        shape_changer._points_seen = 10
        
        with TemporaryDirectory() as tmp_dir:
            shape_changer.save(tmp_dir)

            shape_changer = SlidingWindowReconstruct(window=-1, stride=-1).load(tmp_dir)
            
            self.assertEqual(10, shape_changer.points_seen)
    
    def test_shape_change(self):
        for window, stride, series in itertools.product([1, 2, 3, 7, 10], [1, 2, 3], [self.series_uni, self.series_multi]):
            print(f"Trying sliding window reconstruction with window={window}, stride={stride}")
            shape_changer = SlidingWindowReconstruct(window=window,
                                                     stride=stride)
            
            new_x, new_y = shape_changer.shape_change(series)
            self.assertEqual(series.shape[0], shape_changer.points_seen)
            self.assertEqual(series.ndim + 1, new_x.ndim)
            self.assertEqual(series.ndim + 1, new_y.ndim)
            self.assertEqual((series.shape[0] - window) // stride + 1, new_x.shape[0])
            self.assertEqual(window, new_x.shape[1])
            self.assertEqual(series.shape[1], new_x.shape[2])
            self.assertTupleEqual(new_x.shape, new_y.shape)
            
            for i in range(new_x.shape[0]):
                start = i * stride
                end = start + window
                np.testing.assert_array_equal(series[start:end], new_x[i])
                np.testing.assert_array_equal(series[start:end], new_y[i])
