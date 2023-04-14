from os import PathLike
from os.path import exists
from typing import Any
import json


def save_py_json(obj_to_save,
                 path: PathLike | str) -> None:
    """Save a python object to file using json.
    
    Parameters
    ----------
    obj_to_save : object
        Python object to save on a json file.
    
    path : PathLike
        Path where to store the object.

    Returns
    -------
    None
    """
    json_string = json.JSONEncoder().encode(obj_to_save)
    
    with open(path, mode="w", encoding="utf-8") as file_:
        json.dump(json_string, file_)


def load_py_json(path: PathLike | str) -> Any | None:
    """Load a python object from file saved with json.
    
    Parameters
    ----------
    path : PathLike
        Path where the object is stored.

    Returns
    -------
    result : Any or None
        The object that has been loaded from file or None in case the path does
        not exist.
    """
    if exists(path):
        with open(path, encoding="utf-8") as file_:
            json_string = json.load(file_)
        
        return json.JSONDecoder().decode(json_string)
    else:
        return None
