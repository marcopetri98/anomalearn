import time
import unittest

import numpy as np

from anomalearn.utils import mov_avg, mov_std


def get_windows(series: np.ndarray, window: int, mode: str) -> list[list]:
    if window == 1:
        return [series]
        
    if mode == "left":
        left = window // 2
        right = left - 1 if window % 2 == 0 else left
    else:
        right = window // 2
        left = right - 1 if window % 2 == 0 else right
    left = left
    right = right
    windows = []
    for i in range(series.shape[0]):
        first = i - left if i - left >= 0 else 0
        last = i + 1 + right
        windows.append(series[first:last, :])
    return windows


class TestMovingFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.series1 = np.random.rand(100, 1)
        self.series2 = np.random.rand(100, 4)

    def assert_equal_by_elem_numpy(self, arr1: np.ndarray, arr2: np.ndarray):
        resolution = np.finfo(arr2.dtype).resolution
        for i in range(arr1.shape[0]):
            for j in range(arr1.shape[1]):
                self.assertGreaterEqual(resolution, np.abs(arr1[i, j] - arr2[i, j]))
    
    def test_mov_avg(self):
        for series in [self.series1, self.series2]:
            for window in [1, 2, 5, 10, 100]:
                windows_left = get_windows(series, window, "left")
                windows_right = get_windows(series, window, "right")
                if window != 1:
                    avg_left = np.concatenate([np.mean(e, axis=0) for e in windows_left]).reshape(series.shape)
                    avg_right = np.concatenate([np.mean(e, axis=0) for e in windows_right]).reshape(series.shape)
                else:
                    avg_left = windows_left[0]
                    avg_right = windows_right[0]
                
                mov_avg_left = mov_avg(series, window, "right")
                mov_avg_right = mov_avg(series, window, "left")
                self.assertTupleEqual(avg_left.shape, mov_avg_left.shape)
                self.assertTupleEqual(avg_right.shape, mov_avg_right.shape)
                
                self.assert_equal_by_elem_numpy(avg_left, mov_avg_left)
                self.assert_equal_by_elem_numpy(avg_right, mov_avg_right)
    
    def test_mov_std(self):
        for series in [self.series1, self.series2]:
            for window in [1, 2, 5, 10, 100]:
                windows_left = get_windows(series, window, "left")
                windows_right = get_windows(series, window, "right")
                if window != 1:
                    std_left = np.concatenate([np.std(e, axis=0) for e in windows_left]).reshape(series.shape)
                    std_right = np.concatenate([np.std(e, axis=0) for e in windows_right]).reshape(series.shape)
                else:
                    std_left = np.zeros_like(windows_left[0])
                    std_right = np.zeros_like(windows_right[0])
                
                mov_std_left = mov_std(series, window, "right")
                mov_std_right = mov_std(series, window, "left")
                self.assertTupleEqual(std_left.shape, mov_std_left.shape)
                self.assertTupleEqual(std_right.shape, mov_std_right.shape)
                
                self.assert_equal_by_elem_numpy(std_left, mov_std_left)
                self.assert_equal_by_elem_numpy(std_right, mov_std_right)
                
    def test_speed(self):
        big_series = np.random.rand(50000, 50)

        # make numba compile the functions
        _ = mov_avg(np.random.rand(100, 3), 1, "right")
        _ = mov_std(np.random.rand(100, 3), 1, "right")
        
        for window in [2, 5, 10, 100, 1000]:
            start_time = time.time()
            _ = mov_avg(big_series, window, "right")
            end_time = time.time()
            print(f"Moving average with window {window} of series of shape {big_series.shape}")
            print(f"\tTime elapsed: {end_time - start_time}. With right cut.")
    
            start_time = time.time()
            _ = mov_avg(big_series, window, "left")
            end_time = time.time()
            print(f"\tTime elapsed: {end_time - start_time}. With left cut.")
    
            start_time = time.time()
            _ = mov_std(big_series, window, "right")
            end_time = time.time()
            print(f"Moving standard deviation with window {window} of series of shape {big_series.shape}")
            print(f"\tTime elapsed: {end_time - start_time}. With right cut.")
    
            start_time = time.time()
            _ = mov_std(big_series, window, "left")
            end_time = time.time()
            print(f"\tTime elapsed: {end_time - start_time}. With left cut.", end="\n\n")
