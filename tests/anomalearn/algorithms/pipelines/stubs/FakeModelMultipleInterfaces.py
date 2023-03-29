import numpy as np

from anomalearn.algorithms import IParametric, IPredictor, IClassifier
from anomalearn.algorithms.pipelines import AbstractPipelineBaseLayer
from anomalearn.exceptions import NotTrainedError


class FakeModelMultipleInterfaces(IParametric, IPredictor, IClassifier, AbstractPipelineBaseLayer):
    def __init__(self):
        super().__init__()
        
        self.output_shape = None
        self.called_method = None
    
    def __repr__(self):
        return "FakeModelMultipleInterfaces()"
    
    def __str__(self):
        return "FakeModelMultipleInterfaces"
    
    def __eq__(self, other):
        return isinstance(other, FakeModelMultipleInterfaces)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def get_input_shape(self) -> tuple:
        return tuple(["any shape"])
    
    def get_output_shape(self) -> tuple:
        return tuple(["any shape"])
    
    def copy(self):
        return FakeModelMultipleInterfaces()
    
    def fit(self, x, y=None, *args, **kwargs) -> None:
        self.output_shape = y.shape[1:]
    
    def predict(self, x, *args, **kwargs) -> np.ndarray:
        if self.output_shape is None:
            raise NotTrainedError()
        
        rng = np.random.default_rng(157)
        self.called_method = "predict"
        return rng.random((x.shape[0], *self.output_shape))
    
    def classify(self, x, *args, **kwargs) -> np.ndarray:
        if self.output_shape is None:
            raise NotTrainedError()

        rng = np.random.default_rng(157)
        self.called_method = "classify"
        return rng.random((x.shape[0], *self.output_shape))
