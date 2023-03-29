import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from anomalearn.algorithms import ITransformer, IShapeChanger, ICluster, \
    IParametric, IClassifier, IPredictor, IRegressor
from anomalearn.algorithms.pipelines import Pipeline
from anomalearn.algorithms.postprocessing import BuilderVectorsSlidingWindow, \
    BuilderErrorVectorsDifference, ScorerMahalanobis, ThresholdMaxOnNormal
from anomalearn.algorithms.preprocessing import SlidingWindowReconstruct
from anomalearn.algorithms.transformers import MinMaxScaler, StandardScaler
from anomalearn.exceptions import NotTrainedError
from tests.anomalearn.algorithms.pipelines.stubs.FakeModel import FakeModel
from tests.anomalearn.algorithms.pipelines.stubs.FakeModelMultipleInterfaces import \
    FakeModelMultipleInterfaces


def create_pipeline() -> Pipeline:
    minmax = MinMaxScaler()
    sliding_window = SlidingWindowReconstruct(window=10)
    fake_model = FakeModel()
    vectors_builder = BuilderVectorsSlidingWindow(sliding_window=sliding_window)
    errors_builder = BuilderErrorVectorsDifference()
    scorer = ScorerMahalanobis()
    threshold = ThresholdMaxOnNormal()
    
    return Pipeline([("scaler", minmax, True),
                     ("sliding_window", sliding_window),
                     ("fake_model", fake_model, True),
                     ("vectors_builder", vectors_builder),
                     ("error_builder", errors_builder),
                     ("scorer", scorer, True),
                     ("classification", threshold, True)])


class TestIntegrationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.series_uni = np.random.rand(1000, 1)
        cls.series_multi = np.random.rand(1000, 1)
        
    def setUp(self) -> None:
        self.pipeline = create_pipeline()
        
        self.minmax = self.pipeline.pipeline_layers[0]
        self.sliding_window = self.pipeline.pipeline_layers[1]
        self.fake_model = self.pipeline.pipeline_layers[2]
        self.vectors_builder = self.pipeline.pipeline_layers[3]
        self.errors_builder = self.pipeline.pipeline_layers[4]
        self.scorer = self.pipeline.pipeline_layers[5]
        self.threshold = self.pipeline.pipeline_layers[6]
        
    def test_creation(self):
        self.assertEqual(7, len(self.pipeline))
        self.assertEqual(len(self.pipeline.pipeline_spec), len(self.pipeline.pipeline_names))
        self.assertEqual(len(self.pipeline.pipeline_spec), len(self.pipeline.pipeline_layers))
        self.assertEqual(len(self.pipeline.pipeline_spec), len(self.pipeline.pipeline_train))
        
        self.assertListEqual(["scaler", "sliding_window", "fake_model", "vectors_builder",
                              "error_builder", "scorer", "classification"], self.pipeline.pipeline_names)
        self.assertListEqual([True] * 7, self.pipeline.pipeline_train)

        pipeline = Pipeline([("scaler", self.minmax),
                             ("sliding_window", self.sliding_window),
                             ("fake_model", self.fake_model),
                             ("vectors_builder", self.vectors_builder),
                             ("error_builder", self.errors_builder),
                             ("scorer", self.scorer),
                             ("classification", self.threshold)])
        self.assertListEqual([True] * 7, pipeline.pipeline_train)

        pipeline = Pipeline([(self.minmax, True),
                             (self.sliding_window),
                             (self.fake_model, True),
                             (self.vectors_builder),
                             (self.errors_builder),
                             (self.scorer, True),
                             (self.threshold)])
        self.assertEqual(len(set(self.pipeline.pipeline_names)), len(self.pipeline.pipeline_names))
        self.assertListEqual([True] * 7, pipeline.pipeline_train)
        
        standard_names = []
        for i, e in enumerate(pipeline.pipeline_layers):
            standard_names.append(str(e) + "_" + str(i))
        self.assertListEqual(standard_names, pipeline.pipeline_names)

        for i in range(2):
            pipeline = Pipeline([(self.minmax) if i == 0 else self.minmax,
                                 (self.sliding_window) if i == 0 else self.sliding_window,
                                 (self.fake_model) if i == 0 else self.fake_model,
                                 (self.vectors_builder) if i == 0 else self.vectors_builder,
                                 (self.errors_builder) if i == 0 else self.errors_builder,
                                 (self.scorer) if i == 0 else self.scorer,
                                 (self.threshold) if i == 0 else self.threshold])
            self.assertEqual(len(set(self.pipeline.pipeline_names)), len(self.pipeline.pipeline_names))
            self.assertListEqual([True] * 7, pipeline.pipeline_train)

            standard_names = []
            for i, e in enumerate(pipeline.pipeline_layers):
                standard_names.append(str(e) + "_" + str(i))
            self.assertListEqual(standard_names, pipeline.pipeline_names)
        
    def test_allowed_interfaces(self):
        pipeline = Pipeline([])
        self.assertSetEqual({ITransformer, IShapeChanger, IParametric, ICluster, IClassifier, IRegressor, IPredictor},
                            set(pipeline.allowed_interfaces()))
        
    def test_summary(self):
        self.pipeline.summary()
        
    def test_set_name(self):
        self.pipeline.set_name(0, "Giulio er foggiano")
        self.assertEqual("Giulio er foggiano", self.pipeline.pipeline_names[0])
        
        self.pipeline.set_name("Giulio er foggiano", "John-117")
        self.assertEqual("John-117", self.pipeline.pipeline_names[0])
        
        self.assertRaises(ValueError, self.pipeline.set_name, "sliding_window", "John-117")
        self.assertRaises(IndexError, self.pipeline.set_name, "Giulio er foggiano", "John-117")
        self.assertRaises(IndexError, self.pipeline.set_name, 150, "Giulio er foggiano")
        self.assertRaises(TypeError, self.pipeline.set_name, 15.251, 156132)
        self.assertRaises(TypeError, self.pipeline.set_name, 15.251, "John-117")
        self.assertRaises(TypeError, self.pipeline.set_name, 0, 1254678)
        
    def test_set_trainable(self):
        self.pipeline.set_trainable(0, False)
        self.assertFalse(self.pipeline.pipeline_train[0])

        self.pipeline.set_trainable("scaler", True)
        self.assertTrue(self.pipeline.pipeline_train[0])
        
        self.assertRaises(IndexError, self.pipeline.set_trainable, "Giorgio u veru cavaleri", True)
        self.assertRaises(IndexError, self.pipeline.set_trainable, 150, True)
        self.assertRaises(TypeError, self.pipeline.set_trainable, 15.251, "Giorgio u veru cavaleri")
        self.assertRaises(TypeError, self.pipeline.set_trainable, 15.251, True)
        self.assertRaises(TypeError, self.pipeline.set_trainable, 0, "Giorgio u veru cavaleri")
        
    def test_add_first_layer(self):
        pipeline = create_pipeline()
        
        for addition in range(3):
            previous_len = len(pipeline)
            
            if addition == 0:
                pipeline.add_first_layer(("standard_scaler", StandardScaler(), True))
                self.assertEqual("standard_scaler", pipeline.pipeline_names[0])
            elif addition == 1:
                pipeline.add_first_layer(("standard_scaler_x", StandardScaler()))
                self.assertEqual("standard_scaler_x", pipeline.pipeline_names[0])
            else:
                pipeline.add_first_layer((StandardScaler()))
                self.assertEqual(str(StandardScaler()) + "_9", pipeline.pipeline_names[0])
            
            self.assertEqual(previous_len + 1, len(pipeline))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_names))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_layers))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_train))
            self.assertTrue(pipeline.pipeline_train[0])
            
        # check that a layer with a name equal to an existing layer cannot be inserted
        pipeline = create_pipeline()
        pipeline.add_first_layer(("standard_scaler", StandardScaler(), True))
        self.assertRaises(ValueError, pipeline.add_first_layer, ("standard_scaler", StandardScaler(), True))
    
    def test_insert_layer(self):
        pipeline = create_pipeline()
        
        for addition in range(3):
            previous_len = len(pipeline)
            
            if addition == 0:
                pipeline.insert_layer(3, ("standard_scaler", StandardScaler(), True))
                self.assertEqual("standard_scaler", pipeline.pipeline_names[3])
            elif addition == 1:
                pipeline.insert_layer(3, ("standard_scaler_x", StandardScaler()))
                self.assertEqual("standard_scaler_x", pipeline.pipeline_names[3])
            else:
                pipeline.insert_layer(3, (StandardScaler()))
                self.assertEqual(str(StandardScaler()) + "_9", pipeline.pipeline_names[3])
            
            self.assertEqual(previous_len + 1, len(pipeline))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_names))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_layers))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_train))
            self.assertTrue(pipeline.pipeline_train[3])
            
        # check that a layer with a name equal to an existing layer cannot be inserted
        pipeline = create_pipeline()
        pipeline.insert_layer(3, ("standard_scaler", StandardScaler(), True))
        self.assertRaises(ValueError, pipeline.insert_layer, 3, ("standard_scaler", StandardScaler(), True))
    
    def test_append_layer(self):
        pipeline = create_pipeline()
        
        for addition in range(3):
            previous_len = len(pipeline)
            
            if addition == 0:
                pipeline.append_layer(("standard_scaler", StandardScaler(), True))
                self.assertEqual("standard_scaler", pipeline.pipeline_names[-1])
            elif addition == 1:
                pipeline.append_layer(("standard_scaler_x", StandardScaler()))
                self.assertEqual("standard_scaler_x", pipeline.pipeline_names[-1])
            else:
                pipeline.append_layer((StandardScaler()))
                self.assertEqual(str(StandardScaler()) + "_9", pipeline.pipeline_names[-1])
            
            self.assertEqual(previous_len + 1, len(pipeline))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_names))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_layers))
            self.assertEqual(previous_len + 1, len(pipeline.pipeline_train))
            self.assertTrue(pipeline.pipeline_train[-1])
            
        # check that a layer with a name equal to an existing layer cannot be inserted
        pipeline = create_pipeline()
        pipeline.append_layer(("standard_scaler", StandardScaler(), True))
        self.assertRaises(ValueError, pipeline.append_layer, ("standard_scaler", StandardScaler(), True))
    
    def test_remove_layer(self):
        for removal in range(3):
            pipeline = create_pipeline()
            previous_len = len(pipeline)
            
            if removal == 1:
                layer_name, layer_pos = "scaler", 0
            elif removal == 2:
                layer_name, layer_pos = "vectors_builder", 4
            else:
                layer_name, layer_pos = "classification", -1
            
            removed = pipeline.pipeline_layers[layer_pos]
            pipeline.remove_layer(layer_name)
            self.assertEqual(previous_len - 1, len(pipeline))
            self.assertEqual(previous_len - 1, len(pipeline.pipeline_names))
            self.assertEqual(previous_len - 1, len(pipeline.pipeline_layers))
            self.assertEqual(previous_len - 1, len(pipeline.pipeline_train))
            self.assertNotEqual(layer_name, pipeline.pipeline_names[layer_pos])
            self.assertNotEqual(removed, pipeline.pipeline_layers[layer_pos])
            
            self.assertRaises(ValueError, pipeline.remove_layer, layer_name)
    
    def test_equal(self):
        new_sliding_window = self.sliding_window.copy()
        new_vectors_builder = self.vectors_builder.copy()
        new_vectors_builder._sliding_window = new_sliding_window
        pipeline2 = Pipeline([("scaler", self.minmax.copy(), True),
                              ("sliding_window", new_sliding_window),
                              ("fake_model", self.fake_model.copy(), True),
                              ("vectors_builder", new_vectors_builder),
                              ("error_builder", self.errors_builder.copy()),
                              ("scorer", self.scorer.copy(), True),
                              ("classification", self.threshold.copy())])
        
        copied = create_pipeline()
        
        self.assertEqual(self.pipeline, copied)
        self.assertEqual(self.pipeline, pipeline2)
        
        # check that if two pipeline have different references they are not equal
        sliding_window = SlidingWindowReconstruct(window=10)
        copied.pipeline_layers[3]._sliding_window = sliding_window
        self.assertNotEqual(self.pipeline, copied)
        
        # check that if a pipeline has a different number of layers they are different
        copied = create_pipeline()
        copied.append_layer(sliding_window)
        self.assertNotEqual(self.pipeline, copied)
        
        copied = create_pipeline()
        copied.add_first_layer(sliding_window)
        self.assertNotEqual(self.pipeline, copied)
        
        copied = create_pipeline()
        copied.insert_layer(3, sliding_window)
        self.assertNotEqual(self.pipeline, copied)
        
        copied = create_pipeline()
        copied.remove_layer(3)
        self.assertNotEqual(self.pipeline, copied)
        copied.insert_layer(3, sliding_window)
        self.assertNotEqual(self.pipeline, copied)
        
        # check that a pipeline is equal only to a pipeline
        self.assertNotEqual(self.pipeline, None)
        self.assertNotEqual(self.pipeline, "Etion një mik i vërtetë")
        self.assertNotEqual(self.pipeline, 1789)
    
    def test_identity(self):
        other = create_pipeline()

        self.assertEqual(self.pipeline, other)
        self.assertTrue(self.pipeline.identical(other, degree=2))
        self.assertTrue(self.pipeline.identical(other, degree=3))
        
        other._elements[0] = ("Hajsen best", other._elements[0][1], other._elements[0][2])
        self.assertEqual(self.pipeline, other)
        self.assertTrue(self.pipeline.identical(other, degree=2))
        self.assertFalse(self.pipeline.identical(other, degree=3))
        
        other._elements[0] = ("Cristian friends", other._elements[0][1], False)
        self.assertEqual(self.pipeline, other)
        self.assertFalse(self.pipeline.identical(other, degree=2))
        self.assertFalse(self.pipeline.identical(other, degree=3))
        
    def test_copy(self):
        copied = self.pipeline.copy()
        
        # check that the copy is not shallow
        self.assertIsNot(self.pipeline, copied)
        for layer_1, layer_2 in zip(self.pipeline.pipeline_layers, copied.pipeline_layers):
            self.assertIsNot(layer_1, layer_2)
        
        self.assertEqual(self.pipeline, copied)
        self.assertTrue(self.pipeline.identical(copied, degree=2))
        self.assertTrue(self.pipeline.identical(copied, degree=3))
    
    def test_save_and_load(self):
        with TemporaryDirectory() as tmp_dir:
            self.pipeline.save(tmp_dir)
            
            dirs = [e for e in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, e))]
            files = [e for e in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, e))]
            self.assertEqual(9, len(dirs) + len(files))
            self.assertEqual(5, len(dirs))
            self.assertEqual(4, len(files))
            
            new = Pipeline([])
            self.assertNotEqual(self.pipeline, new)
            
            new.load(tmp_dir, estimator_classes=[FakeModel])
            self.assertTrue(self.pipeline.identical(new, degree=3))
    
    def test_get_hyperparameters(self):
        hyper = self.pipeline.get_hyperparameters()
        
        self.assertEqual(len(self.pipeline), len(hyper))
        self.assertSetEqual(set(self.pipeline.pipeline_names), set(hyper.keys()))
        
        for layer, layer_hyper in hyper.items():
            pos = self.pipeline.pipeline_names.index(layer)
            self.assertDictEqual(self.pipeline.pipeline_layers[pos].get_hyperparameters(), layer_hyper)
    
    def test_set_hyperparameters(self):
        new_pipeline = self.pipeline.copy()
        
        previous_hyper = new_pipeline.get_hyperparameters()
        new_pipeline.set_hyperparameters({"sliding_window": {"window": 50, "stride": 5}})
        new_hyper = new_pipeline.get_hyperparameters()
        
        self.assertNotEqual(self.pipeline, new_pipeline)
        self.assertRaises(AssertionError, self.assertDictEqual, previous_hyper, new_hyper)
        
        previous_hyper_c = previous_hyper.copy()
        new_hyper_c = new_hyper.copy()
        del previous_hyper_c["sliding_window"]
        del new_hyper_c["sliding_window"]
        self.assertDictEqual(previous_hyper_c, new_hyper_c)
    
    def test_fit(self):
        def check_fitted(pipeline):
            self.assertFalse(pipeline.pipeline_train[0])
            self.assertFalse(pipeline.pipeline_train[2])
            self.assertFalse(pipeline.pipeline_train[-1])
            self.assertFalse(pipeline.pipeline_train[-2])
    
            self.assertIsNotNone(pipeline.pipeline_layers[0].seen_data_min)
            self.assertIsNotNone(pipeline.pipeline_layers[-1].threshold)
            self.assertIsNotNone(pipeline.pipeline_layers[-2].mean)

        for series in [self.series_uni, self.series_multi]:
            self.pipeline = create_pipeline()
            self.pipeline.fit(series)
            check_fitted(pipeline=self.pipeline)
            
            with TemporaryDirectory() as tmp_dir:
                self.pipeline = create_pipeline()
                self.pipeline.fit(series, checkpoint_folder=tmp_dir)
                check_fitted(pipeline=self.pipeline)
                
                self.assertEqual(4, len(os.listdir(tmp_dir)))
                for elem in os.listdir(tmp_dir):
                    elem_obj = Path(tmp_dir) / elem
                    self.assertTrue(elem_obj.is_dir())
    
    def test_process(self):
        def create_multiple_interface() -> FakeModelMultipleInterfaces:
            fake_model = FakeModelMultipleInterfaces()
            self.pipeline = create_pipeline()
            self.pipeline.remove_layer(2)
            self.pipeline.insert_layer(2, ("fake_model", fake_model, True))
            return fake_model
        
        def get_pipeline_output_from_layers(series, multiple_int=False):
            x = series
            y = None
            for i in range(len(self.pipeline)):
                layer = self.pipeline.pipeline_layers[i]
                
                if i in [1, 3, 4, 5]:
                    x, y = layer.shape_change(x, y)
                elif i in [0, 6]:
                    x = layer.transform(x)
                else:
                    if multiple_int:
                        if layer.get_pipeline_class() is IPredictor:
                            x = layer.predict(x)
                        else:
                            x = layer.classify(x)
                    else:
                        x = layer.predict(x)
            
            return x

        def create_pipeline_of_pipelines() -> Pipeline:
            minmax = MinMaxScaler()
            sliding_window = SlidingWindowReconstruct(window=10)
            fake_model = FakeModel()
            vectors_builder = BuilderVectorsSlidingWindow(sliding_window=sliding_window)
            errors_builder = BuilderErrorVectorsDifference()
            scorer = ScorerMahalanobis()
            threshold = ThresholdMaxOnNormal()
            pipeline1 = Pipeline([("scaler", minmax, True),
                                  ("sliding_window", sliding_window)])
            pipeline2 = Pipeline([("vectors_builder", vectors_builder),
                                  ("error_builder", errors_builder),
                                  ("scorer", scorer, True),
                                  ("classification", threshold, True)])
            final_pipeline = Pipeline([pipeline1, fake_model, pipeline2])
            return final_pipeline
        
        for series in [self.series_uni, self.series_multi]:
            self.pipeline = create_pipeline()
            self.assertRaises(NotTrainedError, self.pipeline.process, series)
            
            self.pipeline.fit(series)
            output, _ = self.pipeline.process(series)
            self.assertEqual(series.shape[0], output.shape[0])
            desired_output = get_pipeline_output_from_layers(series)
            np.testing.assert_array_equal(desired_output, output)
            
            fake_model = create_multiple_interface()
            self.assertRaises(NotTrainedError, self.pipeline.process, series)

            fake_model.set_pipeline_class(IPredictor)
            self.pipeline.fit(series)
            output, _ = self.pipeline.process(series)
            self.assertEqual("predict", fake_model.called_method)
            self.assertEqual(series.shape[0], output.shape[0])
            desired_output = get_pipeline_output_from_layers(series, True)
            np.testing.assert_array_equal(desired_output, output)

            fake_model.set_pipeline_class(IClassifier)
            self.pipeline.fit(series)
            output, _ = self.pipeline.process(series)
            self.assertEqual("classify", fake_model.called_method)
            self.assertEqual(series.shape[0], output.shape[0])
            desired_output = get_pipeline_output_from_layers(series, True)
            np.testing.assert_array_equal(desired_output, output)
            
        create_pipeline_of_pipelines().summary()
        
        for series in [self.series_uni, self.series_multi]:
            self.pipeline = create_pipeline()
            pipeline_of_pipelines = create_pipeline_of_pipelines()
            self.assertRaises(NotTrainedError, self.pipeline.process, series)
            self.assertRaises(NotTrainedError, pipeline_of_pipelines.process, series)
            
            self.pipeline.fit(series)
            pipeline_of_pipelines.fit(series)
            bare_pipe_out, _ = self.pipeline.process(series)
            pipe_of_pipe_out, _ = pipeline_of_pipelines.process(series)
            np.testing.assert_array_equal(bare_pipe_out, pipe_of_pipe_out)
