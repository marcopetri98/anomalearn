from pathlib import Path
from typing import Any
import importlib
import logging

from ..utils import load_py_json


__module_logger = logging.getLogger(__name__)


def _find_class_in_lib(class_name: str) -> Any:
    """Find a class in anomalearn library and returns it.
    
    Parameters
    ----------
    class_name : str
        Name of the class to search in the library and to retrieve.

    Returns
    -------
    class
        The class to be searched in the library. None if the class doesn't exist.
    """
    this_file = Path(__file__)
    desired_package = None
    for package in this_file.parent.glob("**/__init__.py"):
        with open(package, "r", encoding="utf-8") as f:
            source = f.read()
        
        if class_name in source:
            desired_package = package.parent
    
    desired_package_path = []
    if desired_package is not None:
        for part in reversed(desired_package.parts):
            desired_package_path.insert(0, part)
            if part == "anomalearn":
                break
        
        from_part = ""
        for i, pkg in enumerate(desired_package_path):
            if i != 0:
                from_part += "." + pkg
            else:
                from_part = pkg
        from_part += ".__init__"
        
        return getattr(importlib.import_module(from_part), class_name)
    
    return None


def load_estimator(path: str,
                   estimator_classes: list = None,
                   exclusive_list: bool = False) -> Any:
    """Loads a savable anomalearn estimator.
    
    Parameters
    ----------
    path : str
        The path in which the estimator has been saved.
    
    estimator_classes : list, default=None
        A list of class objects subclassing SavableModel. It is used when new
        objects are created using anomalearn API and are placed outside the
        algorithms package. Pass their class objects as a list and the loading
        function will be able to load them automatically.
        
    exclusive_list : bool, default=False
        States whether the loading function must search for estimator only in
        the list passed to it. If False it considers both the list and the
        anomalearn library estimators.

    Returns
    -------
    estimator
        The anomalearn estimator to be loaded from file.
        
    Raises
    ------
    ValueError
        If the path is not a directory or if the directory does not contain a
        valid anomalearn estimator.
    """
    path_obj = Path(path)
    estimator_classes = estimator_classes if estimator_classes is not None else []

    __module_logger.debug(f"path={str(path_obj)}")
    if not path_obj.is_dir():
        raise ValueError("The given path is not a directory.")

    dir_contents = [e.name for e in path_obj.iterdir()]
    __module_logger.debug(f"os.listdir(path)={dir_contents}")
    if not {"savable_model.json", "signature.json"}.issubset(dir_contents):
        raise ValueError("The directory is not a directory of an anomalearn "
                         "estimator.")
    
    signature_json = load_py_json(str(path_obj / "signature.json"))
    signature = signature_json["signature"]
    class_names = [e.__name__ for e in estimator_classes]
    
    estimator = None
    if signature in class_names:
        # if the loaded class is one of the passed ones, load it
        klass = estimator_classes[class_names.index(signature)]
        estimator = klass.load_model(path)
    elif not exclusive_list:
        # otherwise, search for a package containing the estimator
        klass = _find_class_in_lib(signature)
        if klass is not None:
            estimator = klass.load_model(path)
    
    if estimator is None:
        raise ValueError(f"The folder {path} does not contain an anomalearn valid"
                         "estimator.")
    
    return estimator


def instantiate_estimator(estimator_class_name: str,
                          estimator_classes: list = None,
                          exclusive_list: bool = False) -> Any:
    """Instantiate an estimator with default parameters.
    
    Parameters
    ----------
    estimator_class_name : str
        The name of the estimator object to instantiate (i.e., the class name).
    
    estimator_classes : list, default=None
        A list of class objects subclassing SavableModel. It is used when new
        objects are created using anomalearn API and are placed outside the
        algorithms package. Pass their class objects as a list and the
        instantiation function will be able to instantiate them automatically.
        
    exclusive_list : bool, default=False
        States whether the instantiation function must search for estimator only
        in the list passed to it. If False it considers both the list and the
        anomalearn library estimators.

    Returns
    -------
    estimator
        The anomalearn estimator to be loaded from file.
        
    Raises
    ------
    ValueError
        If the class name is not a name of a valid anomalearn estimator.
    """
    estimator_classes = estimator_classes if estimator_classes is not None else []
    class_names = [e.__name__ for e in estimator_classes]
    
    estimator = None
    if estimator_class_name in class_names:
        # if the class to instantiate is one of the passed ones, instantiate it
        klass = estimator_classes[class_names.index(estimator_class_name)]
        estimator = klass()
    elif not exclusive_list:
        # otherwise, search for a package containing the estimator
        klass = _find_class_in_lib(estimator_class_name)
        if klass is not None:
            estimator = klass()

    if estimator is None:
        raise ValueError(f"The folder {estimator_class_name} isn't an anomalearn"
                         "valid estimator.")

    return estimator
