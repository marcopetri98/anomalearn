from __future__ import annotations

import logging

import pandas as pd

from ...input_validation import check_not_default_attributes
from .. import IDataMultipleReader
from . import TSReader


class TSMultipleReader(TSReader, IDataMultipleReader):
    """A time series reader able to read multiple data series at the same time.

    This class is able to read multiple data series in a single call. The time
    series might be from the same dataset as well they might be from different
    datasets.
    """
    
    def __init__(self):
        super().__init__()

        self.__logger = logging.getLogger(__name__)
        self._all_dataframes: list | None = None
        
    def read_multiple(self, paths: list[str],
                      files_format: str = "csv",
                      pandas_args: dict | None = None,
                      *args,
                      **kwargs) -> TSMultipleReader:
        self._all_dataframes = []
        for idx, path in enumerate(paths):
            self.__logger.info(f"reading from {idx}th path {path}")
            self.read(path=path,
                      file_format=files_format,
                      pandas_args=pandas_args,
                      verbose=False)
            self._all_dataframes.append(self._dataset)

        self.select_dataframe(pos=0)
        self.__logger.info("all datasets read")
        
        return self
    
    def select_dataframe(self, pos: int) -> TSMultipleReader:
        """Selects the dataset to be used.
        
        Parameters
        ----------
        pos : int
            The dataset to select to perform single dataset operations.

        Returns
        -------
        self
            Instance to itself to allow call chaining.
            
        Raises
        ------
        IndexError
            If the index is out of bounds.
        
        ValueError
            If the datasets have not been read.
        """
        if self._all_dataframes is None:
            raise ValueError("read datasets before selecting them")
        
        self._dataset = self._all_dataframes[pos]
        return self
    
    def get_all_dataframes(self, *args, **kwargs) -> list[pd.DataFrame]:
        check_not_default_attributes(self, {"_all_dataframes": None})
        return [e.copy(deep=True) for e in self._all_dataframes]
    
    def get_ith_dataframe(self, pos: int,
                          *args,
                          **kwargs) -> pd.DataFrame:
        check_not_default_attributes(self, {"_all_dataframes": None})
        return self._all_dataframes[pos].copy(deep=True)
