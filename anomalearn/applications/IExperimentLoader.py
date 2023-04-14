from __future__ import annotations

from collections.abc import Iterator, MutableSequence
from typing import Sequence
import abc

import pandas as pd

from ..abc import EqualityABC
from ..reader import IDatasetReader


class IExperimentLoader(MutableSequence, EqualityABC):
    """Interface for experiment loaders.
    
    An experiment loader is a mutable sequence capable of containing only
    dataset readers. Two experiment loaders should be considered equal when
    they have the same number of readers, and the reader at position `i` in both
    experiment loaders is of the same class or the reader implements
    `EqualityABC` and it is tested against the other reader and vice versa.
    Moreover, the train-test splits and the series to use for each reader must
    be identical.
    
    Notes
    -----
    The class is called `IExperimentLoader` because an experiment that is being
    presented for research needs two components: the datasets and the models.
    The datasets are typically loaded from file, and models are instantiated and
    then saved once fitted. Therefore, an experiment is composed of two parts:
    the first is data loading and iteration on all data, the second is the
    training and testing the models on data. Therefore, a loader object should
    be an object loading the needed files for the experiment.
    
    Although the naming choice is a bit subjective, we strongly encourage users
    to propose different naming scheme if they found some that is easier to
    understand and that does not conflict with other package modules.
    """
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __contains__(self, item: IDatasetReader | type):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __getitem__(self, item: int):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __setitem__(self, key: int, value: IDatasetReader | tuple[IDatasetReader, Sequence[float] | None, Sequence[int] | None]):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __delitem__(self, key: int):
        raise NotImplementedError
    
    @abc.abstractmethod
    def index(self, value: IDatasetReader | type,
              start: int = 0,
              stop: int | None = None) -> int:
        raise NotImplementedError
    
    @abc.abstractmethod
    def count(self, value: IDatasetReader | type) -> int:
        raise NotImplementedError
    
    @abc.abstractmethod
    def insert(self, index: int,
               value: IDatasetReader | tuple[IDatasetReader, Sequence[float] | None, Sequence[int] | None]) -> None:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def num_series(self) -> int:
        """Gets the overall number of series that can be retrieved by the experiment.
        
        Returns
        -------
        num_of_series : int
            The number of series that can be retrieved.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_series(self, index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get the series at `index` of the experiment loader.
        
        Parameters
        ----------
        index : int
            It is the index of the time series to get. It must be less than
            `num_series` and it can also be negative. If negative it is relative
            to the last series (e.g., `num_series`).

        Returns
        -------
        training : DataFrame
            The dataframe of the training time series.
        
        testing : DataFrame
            The dataframe of the testing time series.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_train_test_split(self, index: int) -> tuple[float, float] | None:
        """Gets the train-test split for the specified index.
        
        Parameters
        ----------
        index : int
            It is the index of the time series to get. It must be less than
            `num_series` and it can also be negative. If negative it is relative
            to the last series (e.g., `num_series`).

        Returns
        -------
        split : tuple[float, float]
            The train-test split to use for the reader at `index`. None if not
            specific split has been set.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_series_to_use(self, index: int) -> list[int] | None:
        """Gets the series to use for the specified index.
        
        Parameters
        ----------
        index : int
            It is the index of the time series to get. It must be less than
            `num_series` and it can also be negative. If negative it is relative
            to the last series (e.g., `num_series`).

        Returns
        -------
        series_to_use : list[int] or None
            The series to use for the reader at `index`. None if all series must
            be used.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def series_iterator(self, reverse: bool = False) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Returns an iterator which iterates over the series of the experiment.
        
        Parameters
        ----------
        reverse : bool, default=False
            Whether the iterator should iterate on series in reverse order.
        
        Returns
        -------
        series_iterator : Iterator[tuple[pd.DataFrame, pd.DataFrame]]
            It is an iterator which iterates over the series of the experiment.
            It iterates on all readers in order and returns the training and
            testing series for each of the series that must be loaded. If the
            iterator is reversed, it iterates from the last reader to the first
            and for each of them from the last series to the first. It yields
            to dataframes.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_train_test_split(self, index: int,
                             split: Sequence[float] | None) -> IExperimentLoader:
        """Set the train-test split for the reader at `index`.
        
        Parameters
        ----------
        index : int
            The position of the reader to which the train-test split must be
            set.
            
        split : Sequence[float] or None
            It is a sequence of length 2 in which the first element is the
            percentage of training points and the second is the percentage of
            the test points. If None, it sets the default.

        Returns
        -------
        self
            Instance to itself.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def set_series_to_use(self, index: int,
                          series: Sequence[int] | None) -> IExperimentLoader:
        """Sets the series to use for reader at `index`.
        
        Parameters
        ----------
        index : int
            The position of the reader to which the number of series to read
            must be set.
        
        series : Sequence[int] or None
            It is a sequence of number of series to load from the reader at
            `index`. If None, it sets the default.

        Returns
        -------
        self
            Instance to itself.
        """
        raise NotImplementedError
