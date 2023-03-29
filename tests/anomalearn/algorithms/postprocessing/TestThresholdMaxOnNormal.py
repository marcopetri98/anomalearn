import unittest
from tempfile import TemporaryDirectory

import numpy as np

from anomalearn.algorithms.postprocessing import ThresholdMaxOnNormal


class TestThresholdMaxOnNormal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scores = np.random.rand(1000)
    
    def test_equality(self):
        threshold1 = ThresholdMaxOnNormal()
        threshold2 = ThresholdMaxOnNormal()
        
        self.assertEqual(threshold1, threshold2)
        
        threshold1._threshold = 10
        
        self.assertNotEqual(threshold1, threshold2)

        threshold2._threshold = 10
        
        self.assertEqual(threshold1, threshold2)
    
    def test_copy(self):
        threshold1 = ThresholdMaxOnNormal()
        threshold2 = threshold1.copy()
        
        self.assertIsNot(threshold1, threshold2)
        self.assertEqual(threshold1, threshold2)
        
        threshold1._threshold = 111
        threshold2 = threshold1.copy()
        self.assertIsNot(threshold1, threshold2)
        self.assertEqual(threshold1, threshold2)
        
    def test_fit(self):
        threshold1 = ThresholdMaxOnNormal()
        self.assertIsNone(threshold1.threshold)
        
        threshold1.fit(self.scores)
        self.assertEqual(np.max(self.scores), threshold1.threshold)
        
    def test_transform(self):
        threshold = ThresholdMaxOnNormal()
        threshold.fit(self.scores)
        
        self.assertEqual(0, np.sum(threshold.transform(self.scores)))
        
        new_series = self.scores.copy()
        new_series[:10] = 10
        self.assertEqual(10, np.sum(threshold.transform(new_series)))

    def test_save_and_load(self):
        threshold = ThresholdMaxOnNormal()
        
        with TemporaryDirectory() as tmp_dir:
            threshold.save(tmp_dir)
            loaded = ThresholdMaxOnNormal().load(tmp_dir)
            self.assertIsNot(loaded, threshold)
            self.assertEqual(loaded, threshold)
            
            threshold._threshold = "Altair Ibn-La'Ahad"
            threshold.save(tmp_dir)
            loaded = ThresholdMaxOnNormal().load(tmp_dir)
            self.assertIsNot(loaded, threshold)
            self.assertEqual(loaded, threshold)
            
