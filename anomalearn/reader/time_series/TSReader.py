from __future__ import annotations

import logging
import os.path

import pandas as pd

from ...input_validation.attribute_checks import check_not_default_attributes
from .. import IDataReader


class TSReader(IDataReader):
    """A reader of time series datasets."""
    accepted_formats = ["csv", "json", "xml"]
    
    def __init__(self):
        super().__init__()
        
        self.__logger = logging.getLogger(__name__)
        self._dataset: pd.DataFrame | None = None
    
    def read(self, path,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> TSReader:
        if not os.path.isfile(path):
            raise ValueError(f"The file path \"{path}\" you are trying to read "
                             "does not exists.")
        elif pandas_args is not None and not isinstance(pandas_args, dict):
            raise TypeError("pandas_args must be None or a dict")

        pandas_args = pandas_args if pandas_args is not None else {}

        match file_format:
            case "csv":
                self.__logger.debug(f"reading using pd.read_csv from {path}")
                self._dataset = pd.read_csv(path, **pandas_args)

            case "json":
                self.__logger.debug(f"reading using pd.read_json from {path}")
                self._dataset = pd.read_json(path, **pandas_args)

            case "xml":
                self.__logger.debug(f"reading using pd.read_xml from {path}")
                self._dataset = pd.read_xml(path, **pandas_args)

            case _:
                raise NotImplementedError("The dataset format is not supported,"
                                          " the accepted formats are "
                                          f"{self.accepted_formats}")
        
        self.__logger.info(f"dataset read from {path}")
        
        return self
    
    def get_dataframe(self, *args, **kwargs) -> pd.DataFrame:
        check_not_default_attributes(self, {"_dataset": None})
        return self._dataset.copy(deep=True)
