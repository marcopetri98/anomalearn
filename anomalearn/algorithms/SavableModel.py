from __future__ import annotations

from pathlib import Path

from .. import EqualityABC, FullyRepresentableABC
from ..utils import find_or_create_dir, load_py_json, save_py_json
from . import BaseModel, ISavable


class SavableModel(ISavable, FullyRepresentableABC, EqualityABC, BaseModel):
    """Object representing a base model that can be saved.
    
    If an object can be saved, there are either parameters, hyperparameters or
    configuration parameters that can be saved. If none of the previous is
    present, a model must not be savable.
    """
    __json_file = "savable_model.json"
    __json_signature = "signature.json"
        
    def __repr__(self):
        return "SavableModel()"
    
    def __str__(self):
        return "SavableModel"
    
    def __eq__(self, other):
        return isinstance(other, SavableModel)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def save(self, path,
             *args,
             **kwargs) -> SavableModel:
        find_or_create_dir(path)
        path_obj = Path(path)
        
        json_objects = self.get_params(deep=False)
        save_py_json(json_objects, str(path_obj / self.__json_file))
        
        save_py_json({"signature": self.__class__.__name__}, str(path_obj / self.__json_signature))
        
        return self

    def load(self, path: str,
             *args,
             **kwargs) -> SavableModel:
        path_obj = Path(path)

        if not path_obj.joinpath(self.__json_file).is_file():
            raise ValueError("path directory is not valid. It must contain "
                             f"these files: {self.__json_file}, {self.__json_signature}")

        json_objects: dict = load_py_json(str(path_obj / self.__json_file))
        
        self.set_params(**json_objects)
        
        return self
    
    @classmethod
    def load_model(cls, path: str,
                   *args,
                   **kwargs) -> SavableModel:
        """Loads the saved object from a folder.
        
        Parameters
        ----------
        path : str
            It is the path of the directory in which the object has been saved.

        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        model
            The instance of the saved object.
        """
        obj = SavableModel()
        obj.load(path)
        return obj
