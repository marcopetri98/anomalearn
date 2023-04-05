import unittest
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.ensemble import IsolationForest as sklearnIsolationForest

from anomalearn.algorithms.models.machine_learning import IsolationForest


class TestIsolationForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(100, 1)
        cls.series_multi = np.random.rand(100, 5)
    
    def test_fit(self):
        isof = IsolationForest()
        
        self.assertIsNone(isof.base_estimator)
        self.assertIsNone(isof.estimators)
        self.assertIsNone(isof.estimators_features)
        self.assertIsNone(isof.estimators_samples)
        self.assertIsNone(isof.seen_max_samples)
        self.assertIsNone(isof.offset)
        self.assertIsNone(isof.seen_features_in)
        self.assertIsNone(isof.seen_feature_names_in)
        
        for series in [self.series_uni, self.series_multi]:
            isof.fit(series)

            self.assertIsNotNone(isof.base_estimator)
            self.assertIsNotNone(isof.estimators)
            self.assertIsNotNone(isof.estimators_features)
            self.assertIsNotNone(isof.estimators_samples)
            self.assertIsNotNone(isof.seen_max_samples)
            self.assertIsNotNone(isof.offset)
            self.assertIsNotNone(isof.seen_features_in)
            
            self.assertEqual(series.shape[1], isof.seen_features_in)
    
    def test_classify(self):
        isof = IsolationForest(random_state=11)
        sklearn_isof = sklearnIsolationForest(random_state=11)
        
        for series in [self.series_uni, self.series_multi]:
            isof.fit(series)
            sklearn_isof.fit(series)
            pred = isof.classify(series)
            sklearn_pred = sklearn_isof.predict(series)
            
            np.testing.assert_array_equal(pred, sklearn_pred)
    
    def test_decision_function(self):
        isof = IsolationForest(random_state=11)
        sklearn_isof = sklearnIsolationForest(random_state=11)
        
        for series in [self.series_uni, self.series_multi]:
            isof.fit(series)
            sklearn_isof.fit(series)
            pred = isof.decision_function(series)
            sklearn_pred = sklearn_isof.decision_function(series)
            
            np.testing.assert_array_equal(pred * -1, sklearn_pred)
    
    def test_anomaly_score(self):
        isof = IsolationForest(random_state=11)
        sklearn_isof = sklearnIsolationForest(random_state=11)
        
        for series in [self.series_uni, self.series_multi]:
            isof.fit(series)
            sklearn_isof.fit(series)
            pred = isof.anomaly_score(series)
            sklearn_pred = sklearn_isof.score_samples(series)
            
            np.testing.assert_array_equal(pred * -1, sklearn_pred)
    
    def test_equality(self):
        isof1 = IsolationForest(random_state=11)
        isof2 = IsolationForest(random_state=11)
        
        self.assertEqual(isof1, isof2)
        self.assertEqual(isof2, isof1)
        
        isof1.fit(self.series_uni)
        self.assertNotEqual(isof1, isof2)
        self.assertNotEqual(isof2, isof1)
        
        isof2.fit(self.series_uni)
        self.assertEqual(isof1, isof2)
        self.assertEqual(isof2, isof1)
    
    def test_copy(self):
        isof1 = IsolationForest(random_state=11)
        
        isof2 = isof1.copy()
        self.assertEqual(isof1, isof2)
        self.assertIsNot(isof1, isof2)
        
        isof1.fit(self.series_uni)
        isof2 = isof1.copy()
        self.assertEqual(isof1, isof2)
        self.assertIsNot(isof1, isof2)
    
    def test_save_and_load(self):
        isof = IsolationForest(random_state=11)
        
        with TemporaryDirectory() as tmp_dir:
            isof.save(tmp_dir)
            
            new_isof = IsolationForest()
            new_isof.fit(self.series_uni)
            new_isof.load(tmp_dir)
            new_isof2 = IsolationForest.load_model(tmp_dir)
            self.assertIsNone(new_isof.seen_features_in)
            self.assertEqual(isof, new_isof)
            self.assertEqual(isof, new_isof2)
            
            isof.fit(self.series_multi)
            isof.save(tmp_dir)
            new_isof.load(tmp_dir)
            new_isof2 = IsolationForest.load_model(tmp_dir)
            self.assertEqual(isof, new_isof)
            self.assertEqual(isof, new_isof2)
