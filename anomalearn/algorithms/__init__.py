from .BaseModel import BaseModel

from .IClassifier import IClassifier
from .ICluster import ICluster
from .ICopyable import ICopyable
from .IParametric import IParametric
from .IMultipleParametric import IMultipleParametric
from .IPredictor import IPredictor
from .IRegressor import IRegressor
from .IShapeChanger import IShapeChanger
from .ITransformer import ITransformer

from .ISavable import ISavable
from .SavableModel import SavableModel

from .algo_functions import load_estimator, instantiate_estimator
