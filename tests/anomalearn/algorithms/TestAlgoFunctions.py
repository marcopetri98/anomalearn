import unittest
from tempfile import TemporaryDirectory

from anomalearn.algorithms import SavableModel, load_estimator, \
    instantiate_estimator
from anomalearn.algorithms.pipelines import Pipeline
from anomalearn.algorithms.postprocessing import \
    BuilderErrorVectorsAbsDifference, BuilderErrorVectorsDifference
from anomalearn.algorithms.preprocessing import SlidingWindowReconstruct


class TestAlgoFunctions(unittest.TestCase):
    def test_load_estimator(self):
        savable_model = SavableModel()
        savable_pipeline = Pipeline([])
        savable_sliding_window = SlidingWindowReconstruct()
        
        with TemporaryDirectory() as tmp_dir:
            savable_model.save(tmp_dir)
            loaded = load_estimator(tmp_dir)
            self.assertIsInstance(loaded, SavableModel)
            self.assertEqual(savable_model, loaded)
            
            self.assertRaises(ValueError, load_estimator, tmp_dir, [Pipeline], True)
            self.assertRaises(ValueError, load_estimator, tmp_dir, [SlidingWindowReconstruct], True)
            
            savable_pipeline.save(tmp_dir)
            loaded = load_estimator(tmp_dir)
            self.assertIsInstance(loaded, Pipeline)
            self.assertEqual(savable_pipeline, loaded)
            
            self.assertRaises(ValueError, load_estimator, tmp_dir, [SlidingWindowReconstruct], True)
            
            savable_sliding_window.save(tmp_dir)
            loaded = load_estimator(tmp_dir)
            self.assertIsInstance(loaded, SlidingWindowReconstruct)
            self.assertEqual(savable_sliding_window, loaded)
            
            self.assertRaises(ValueError, load_estimator, tmp_dir, [BuilderErrorVectorsDifference], True)
            self.assertRaises(ValueError, load_estimator, tmp_dir, [BuilderErrorVectorsAbsDifference], True)
    
    def test_instantiate_estimator(self):
        class1 = "BuilderErrorVectorsAbsDifference"
        class2 = "BuilderErrorVectorsDifference"
        class3 = "SlidingWindowReconstruct"
        
        model1 = instantiate_estimator(class1)
        self.assertIsInstance(model1, BuilderErrorVectorsAbsDifference)
        self.assertRaises(ValueError, instantiate_estimator, class1, [SlidingWindowReconstruct], True)
        
        model2 = instantiate_estimator(class2)
        self.assertIsInstance(model2, BuilderErrorVectorsDifference)
        self.assertRaises(ValueError, instantiate_estimator, class2, [SlidingWindowReconstruct], True)
        
        model3 = instantiate_estimator(class3)
        self.assertIsInstance(model3, SlidingWindowReconstruct)
        self.assertRaises(ValueError, instantiate_estimator, class3, [BuilderErrorVectorsDifference], True)
        
