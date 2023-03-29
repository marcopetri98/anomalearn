from __future__ import annotations

import abc
from abc import ABC

import pandas as pd


class IDataReader(ABC):
    """Interface for all dataset readers in the repository.
    """
    
    @abc.abstractmethod
    def read(self, path,
             file_format: str,
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> IDataReader:
        """Reads a dataset and returns an instance of itself.

        When the dataset is read, a dataframe to represent the dataset is built
        to represent it.
        
        Parameters
        ----------
        path
            It is a path representing the location on disk of the dataset to
            read.
        
        file_format : ["csv", "json"]
            It is the format in which the dataset is stored.

        pandas_args : dict or None
            This dict represent all the additional params to be used while
            reading the dataset with pandas. The params depend on the file
            format of the dataset. If the format is "csv", the additional params
            will be the pandas params for `read_csv` and so on.
            
        args
            Not used, present to allow multiple inheritance and signature change.
            
        kwargs
            Not used, present to allow multiple inheritance and signature change.

        Returns
        -------
        IDataReader
            Instance of itself to be able to chain calls.
            
        Raises
        ------
        NotImplementedError
            If the file format is not supported.
        """
        pass
    
    @abc.abstractmethod
    def get_dataframe(self, *args, **kwargs) -> pd.DataFrame:
        """Gets the dataframe of the dataset previously read.
        
        Parameters
        ----------
        args
            Not used, present to allow multiple inheritance and signature change.

        kwargs
            Not used, present to allow multiple inheritance and signature change.
        
        Returns
        -------
        dataframe : DataFrame
            The dataframe of the dataset.
            
        Raises
        ------
        ValueError
            If the dataset has not been read.
        """
        pass
