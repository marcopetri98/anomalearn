import logging
from pathlib import Path


__module_logger = logging.getLogger(__name__)


def find_or_create_dir(path: str) -> None:
    """Find or create the dir at the specified path.
    
    Parameters
    ----------
    path : str
        The path of the dir to search and eventually create.

    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the given path points to a file and not to a directory.
    """
    path_obj = Path(path)
    __module_logger.debug(f"path={str(path_obj)}")

    if not path_obj.is_dir() and path_obj.exists():
        raise ValueError("path must point to an existing directory or to a "
                         "non existing directory")

    # if the directory does not exist, create it
    if not path_obj.exists():
        path_obj.mkdir(parents=True)
