import unittest
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.preprocessing import MinMaxScaler as scikitMinMaxScaler

from anomalearn.algorithms.transformers import MinMaxScaler


class TestMinMaxScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(100, 1)
        cls.series_multi = np.random.rand(100, 5)
    
    def test_fit(self):
        scaler = MinMaxScaler()

        self.assertIsNone(scaler.seen_data_min)
        self.assertIsNone(scaler.seen_data_max)
        self.assertIsNone(scaler.seen_data_range)
        self.assertIsNone(scaler.seen_features_in)
        self.assertIsNone(scaler.seen_samples_in)
        
        for series in [self.series_uni, self.series_multi]:
            scaler.fit(series)
            
            self.assertIsNotNone(scaler.seen_data_min)
            self.assertIsNotNone(scaler.seen_data_max)
            self.assertIsNotNone(scaler.seen_data_range)
            self.assertIsNotNone(scaler.seen_features_in)
            self.assertIsNotNone(scaler.seen_samples_in)
            self.assertEqual(series.shape[1], scaler.seen_features_in)
    
    def test_transform(self):
        scaler = MinMaxScaler()
        
        for series in [self.series_uni, self.series_multi]:
            scaler.fit(series)
            out = scaler.transform(series)
            scikit_out = scikitMinMaxScaler()
            
            np.testing.assert_array_equal(scikit_out.fit_transform(series), out)
    
    def test_save_and_load(self):
        scaler = MinMaxScaler()
        
        with TemporaryDirectory() as tmp_dir:
            scaler.save(tmp_dir)
            
            new_scaler = MinMaxScaler()
            new_scaler.fit(self.series_uni)
            new_scaler.load(tmp_dir)
            self.assertIsNone(new_scaler.seen_data_min)
            
            scaler.fit(self.series_uni)
            scaler.save(tmp_dir)
            
            new_scaler = MinMaxScaler()
            new_scaler.load(tmp_dir)
            self.assertIsNotNone(new_scaler.seen_data_min)
    
    def test_copy(self):
        scaler = MinMaxScaler()
        new = scaler.copy()
        
        self.assertIsNot(scaler, new)
        self.assertIsNot(scaler._min_max_scaler, new._min_max_scaler)
        
        scaler.fit(self.series_uni)
        new = scaler.copy()
        
        self.assertIsNot(scaler, new)
        self.assertIsNot(scaler._min_max_scaler, new._min_max_scaler)
        
        self.assertIsNot(scaler.seen_data_min, new.seen_data_min)
        self.assertIsNot(scaler.seen_data_max, new.seen_data_max)
        self.assertIsNot(scaler.seen_data_range, new.seen_data_range)
        self.assertEqual(scaler.seen_samples_in, new.seen_samples_in)
        self.assertEqual(scaler.seen_features_in, new.seen_features_in)
        np.testing.assert_array_equal(scaler.seen_data_min, new.seen_data_min)
        np.testing.assert_array_equal(scaler.seen_data_max, new.seen_data_max)
        np.testing.assert_array_equal(scaler.seen_data_range, new.seen_data_range)
