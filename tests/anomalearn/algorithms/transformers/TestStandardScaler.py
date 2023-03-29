import unittest
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.preprocessing import StandardScaler as scikitStandardScaler

from anomalearn.algorithms.transformers import StandardScaler


class TestStandardScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(100, 1)
        cls.series_multi = np.random.rand(100, 5)
    
    def test_fit(self):
        scaler = StandardScaler()

        self.assertIsNone(scaler.seen_mean)
        self.assertIsNone(scaler.seen_var)
        self.assertIsNone(scaler.seen_scale)
        self.assertIsNone(scaler.seen_features_in)
        self.assertIsNone(scaler.seen_samples_in)
        
        for series in [self.series_uni, self.series_multi]:
            scaler.fit(series)
            
            self.assertIsNotNone(scaler.seen_mean)
            self.assertIsNotNone(scaler.seen_var)
            self.assertIsNotNone(scaler.seen_scale)
            self.assertIsNotNone(scaler.seen_features_in)
            self.assertIsNotNone(scaler.seen_samples_in)
            self.assertEqual(series.shape[1], scaler.seen_features_in)
    
    def test_transform(self):
        scaler = StandardScaler()
        
        for series in [self.series_uni, self.series_multi]:
            scaler.fit(series)
            out = scaler.transform(series)
            scikit_out = scikitStandardScaler()
            
            np.testing.assert_array_equal(scikit_out.fit_transform(series), out)
    
    def test_save_and_load(self):
        scaler = StandardScaler()
        
        with TemporaryDirectory() as tmp_dir:
            scaler.save(tmp_dir)
            
            new_scaler = StandardScaler()
            new_scaler.fit(self.series_uni)
            new_scaler.load(tmp_dir)
            self.assertIsNone(new_scaler.seen_mean)
            
            scaler.fit(self.series_uni)
            scaler.save(tmp_dir)
            
            new_scaler = StandardScaler()
            new_scaler.load(tmp_dir)
            self.assertIsNotNone(new_scaler.seen_mean)
    
    def test_copy(self):
        scaler = StandardScaler()
        new = scaler.copy()
        
        self.assertIsNot(scaler, new)
        self.assertIsNot(scaler._standard_scaler, new._standard_scaler)
        
        scaler.fit(self.series_uni)
        new = scaler.copy()
        
        self.assertIsNot(scaler, new)
        self.assertIsNot(scaler._standard_scaler, new._standard_scaler)
        
        self.assertIsNot(scaler.seen_mean, new.seen_mean)
        self.assertIsNot(scaler.seen_var, new.seen_var)
        self.assertIsNot(scaler.seen_scale, new.seen_scale)
        self.assertEqual(scaler.seen_samples_in, new.seen_samples_in)
        self.assertEqual(scaler.seen_features_in, new.seen_features_in)
        np.testing.assert_array_equal(scaler.seen_mean, new.seen_mean)
        np.testing.assert_array_equal(scaler.seen_var, new.seen_var)
        np.testing.assert_array_equal(scaler.seen_scale, new.seen_scale)
