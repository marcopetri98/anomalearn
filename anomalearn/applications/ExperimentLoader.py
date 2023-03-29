import logging
import warnings
from collections.abc import Sequence, Iterable
from copy import copy
from numbers import Number
from typing import Iterator

import pandas as pd

from . import IExperimentLoader
from .. import EqualityABC
from ..reader import IDatasetReader
from ..reader.time_series import rts_config


class ExperimentLoader(IExperimentLoader):
    """An implementation of the interface experiment loader.
    
    Parameters
    ----------
    readers : Sequence[IDatasetReader] or None, default=None
        It is a sequence of dataset readers that must be used in this experiment.
        The order of the sequence matters since they will be read in the exact
        same order of the given sequence. If None, the experiment loader is
        initialized without readers.
    
    train_test_splits : Sequence[tuple[float, float] or None] | None, default=None
        It is a sequence of train-test splits in which the first is the training
        percentage and the second is the test percentage. In such a case, the
        sum of the two elements must be 1. If an element of the list is None it
        means to use the `default_split` or the dataset's training set. If it
        is None, default is always used.
    
    series_to_use : Sequence[Sequence[int] or None] or None, default=None
        It is an optional sequence of time series indices to be used from the
        dataset reader (e.g., an element [1, 2, 3] states that series 1, 2, 3
        must be used for that dataset, other series will be ignored). It an
        element is None means to use all the series in the dataset. If it is
        None, default is always used.
    
    default_split : Sequence[float] or length 2, default=(0.8, 0.2)
        The standard train-test split to perform when the dataset does not
        provide a predefined training set in which the first element is the
        training portion and the second is the testing portion. The sum of the
        two elements must be 1.
        
    Attributes
    ----------
    __logger
        The logger to log messages.
    
    _default_split
        The default split to be used.
    
    _readers
        The list of the readers on the experiment.
    
    _splits
        The list of the splits for the readers of the experiment.
    
    _series
        The series to be used for the readers of the experiment.
    """

    def __init__(self, readers: Sequence[IDatasetReader] | None = None,
                 train_test_splits: Sequence[Sequence[float] | None] | None = None,
                 series_to_use: Sequence[Sequence[int] | None] | None = None,
                 default_split: Sequence[float] = (0.8, 0.2)):
        super().__init__()
        
        self.__logger = logging.getLogger(__name__)
        
        if default_split[0] + default_split[1] != 1:
            raise ValueError(f"default_split={default_split}, it must sum up to 1")
        self._default_split = (default_split[0], default_split[1])
        
        # check and fix some user input problems
        if readers is None:
            self.__logger.info("the experiment does not have readers. Fill it")
            readers = []
            
            if train_test_splits is not None or series_to_use is not None:
                warnings.warn("readers=None implies other should be None too. "
                              "They are now None.",
                              UserWarning)
                train_test_splits = None
                series_to_use = None
        elif train_test_splits is not None and len(train_test_splits) != len(readers):
            warnings.warn("if train_test_split is passed, it should have the "
                          "same length of readers. It is now None")
            train_test_splits = []
        elif series_to_use is not None and len(series_to_use) != len(readers):
            warnings.warn("if series_to_use is passed, it should have the same "
                          "length of readers. It is now None")
            series_to_use = []
            
        # check that readers are correct
        for reader in readers:
            if not isinstance(reader, IDatasetReader):
                raise ValueError(f"readers must be subclasses of "
                                 f"{str(IDatasetReader)}")
        if readers is not None:
            self._readers = copy(readers)
        else:
            self._readers = []

        self._splits = []
        # adjust length of default train test splits
        if len(readers) != 0 and (train_test_splits is None or len(train_test_splits) == 0):
            self._splits = [self._default_split] * len(readers)
        elif len(readers) != 0:
            self._splits = []
            for split in train_test_splits:
                self._splits.append(self.__fix_input_split(split))

        self._series = []
        # adjust the series to be used
        if len(readers) != 0 and (series_to_use is None or len(series_to_use) == 0):
            self._series = [None] * len(readers)
        elif len(readers) != 0:
            self._series = []
            for i, indices in enumerate(series_to_use):
                self._series.append(self.__fix_input_series(self._readers[i], indices))

    @staticmethod
    def __fix_input_split(split: Sequence[float] | None) -> tuple[float, float] | None:
        """Fix the given split to the standard split format.
        
        Parameters
        ----------
        split : Sequence[float] or None
            The split to check.

        Returns
        -------
        standard_split : tuple[float, float]
            The split as a tuple of two floats between 0 and 1 whose sum is 1.
            
        Raises
        ------
        ValueError
            If the split does not have the correct format.
        """
        try:
            if split is not None and (not isinstance(split[0], Number) or not isinstance(split[1], Number)):
                raise TypeError(f"found a split ({split}), it should be a "
                                f"sequence of numbers")
            elif split is not None and split[0] + split[1] != 1:
                raise ValueError(f"found a split ({split}), sum must be 1")
            elif split is not None:
                return split[0], split[1]
            else:
                return None
        except IndexError:
            raise TypeError("splits must be tuple of length two")
        
    @staticmethod
    def __fix_input_series(reader: IDatasetReader,
                           indices: Sequence[int] | None) -> list[int] | None:
        """Fix the given indices to list of indices between 0 and len(reader)-1.
        
        Parameters
        ----------
        reader : IDatasetReader
            The reader to which the time series indices are selected.
        
        indices : Sequence[int] or None
            The sequence of indices for the reader to include in the experiment.

        Returns
        -------
        series_indices : list[int] or None
            The list of the indices of the time series to include in the
            experiment or None if all must be used.
            
        Raises
        ------
        ValueError
            If elements of the list are not integers or if the index is wrong
            and would cause an IndexError on the reader.
        """
        try:
            if indices is not None:
                for index in indices:
                    if not isinstance(index, int):
                        raise TypeError(f"found indices [{indices}], they must "
                                        f"be integers")
                    elif index >= len(reader):
                        raise ValueError(f"invalid index for reader {str(reader)}. "
                                         f"Found index={index}")
                    elif index < 0 and len(reader) - index < 0:
                        raise ValueError(f"invalid index for reader {str(reader)}. "
                                         f"Found index={index}")

                return [e if e >= 0 else len(reader) - e for e in indices]
            else:
                return None
        except TypeError:
            raise TypeError("series to be used must be iterables")

    def __fix_set_insert_input(self, value: IDatasetReader | tuple[IDatasetReader,
                                                                   Sequence[float] | None,
                                                                   Sequence[int] | None]) -> tuple[IDatasetReader,
                                                                                                   tuple[float, float],
                                                                                                   list[int] | None]:
        """Fix the input value and returns triple reader, split, series.
        
        Parameters
        ----------
        value : IDatasetReader | tuple[IDatasetReader, Sequence[float] or None, Sequence[int] or None]
            The input value to fix.

        Returns
        -------
        reader : IDatasetReader
            The reader obtained from `value`.
        
        split : tuple[float]
            The split obtained from `value`.
        
        series : list[int] or None
            The series obtained from `value`.
        """
        if isinstance(value, IDatasetReader):
            reader = value
            split = self._default_split
            series = None
        else:
            if not isinstance(value[0], IDatasetReader):
                raise TypeError(f"the first element is {value[0]}, it should be "
                                f"an IDatasetReader")
            reader = value[0]
            split = self.__fix_input_split(value[1])
            series = self.__fix_input_series(value[0], value[2])
            
        return reader, split, series
    
    @staticmethod
    def __are_readers_equal(r1: IDatasetReader,
                            r2: IDatasetReader | type) -> bool:
        """Checks if the reader `r1` is equal or is of the same class of `r2`.
        
        Parameters
        ----------
        r1 : IDatasetReader
            The reader to be checked against `r2`.
        
        r2 : IDatasetReader or type
            Another reader or a reader class type.

        Returns
        -------
        are_equal : bool
            True if the two can be considered equal, False otherwise.
        """
        if isinstance(r2, IDatasetReader):
            if isinstance(r1, EqualityABC) and not r1 == r2:
                return False
            elif isinstance(r2, EqualityABC) and not r2 == r1:
                return False
            else:
                return isinstance(r1, r2.__class__)
        else:
            return isinstance(r1, r2)
        
    def __check_index(self, index) -> int:
        """Checks that a number is a valid index.
        
        Parameters
        ----------
        index : int
            A positive or negative number representing an index.

        Returns
        -------
        true_index : int
            Positive and valid index.
            
        Raises
        ------
        IndexError
            If the index is wrong.
        """
        if index < 0:
            index = len(self._readers) + index

        if not 0 <= index < len(self._readers):
            raise IndexError("experiment loader assignment index out of range")
        
        return index

    def __str__(self):
        return "ExperimentLoader"

    def __eq__(self, other):
        if not isinstance(other, ExperimentLoader):
            return False
        elif len(other) != len(self):
            return False
        
        for r1, r2 in zip(self._readers, other._readers):
            if not self.__are_readers_equal(r1, r2):
                return False

        for s1, s2 in zip(self._splits, other._splits):
            if s1 != s2:
                return False

        for s1, s2 in zip(self._series, other._series):
            if s1 != s2:
                return False
            
        return True

    def __len__(self):
        return len(self._readers)

    def __contains__(self, item: IDatasetReader | type):
        for reader in self._readers:
            if self.__are_readers_equal(reader, item):
                return True
        else:
            return False

    def __getitem__(self, item: int):
        item = self.__check_index(item)
        reader = self._readers[item]
        split = self._splits[item]
        series = self._series[item]
        return reader, split, copy(series)

    def __setitem__(self,
                    key: int,
                    value: IDatasetReader | Sequence[IDatasetReader, Sequence[float] | None, Sequence[int] | None]):
        index = self.__check_index(key)
        reader, split, series = self.__fix_set_insert_input(value)
        self._readers[index] = reader
        self._splits[index] = split
        self._series[index] = series

    def __delitem__(self, key: int):
        index = self.__check_index(key)
        del self._readers[index]
        del self._splits[index]
        del self._series[index]

    def index(self,
              value: IDatasetReader | type,
              start: int = 0,
              stop: int | None = None) -> int:
        start = max(0, start)
        if stop is None:
            stop = len(self._readers)
        else:
            stop = min(len(self._readers), stop)
        
        for i in range(start, stop):
            if self.__are_readers_equal(self._readers[i], value):
                return i
        else:
            raise ValueError(f"loader does not contain a reader like {str(value)}")

    def count(self, value: IDatasetReader | type) -> int:
        return sum([1 for e in self._readers if self.__are_readers_equal(e, value)])

    def insert(self,
               index: int,
               value: IDatasetReader | tuple[IDatasetReader, Sequence[float] | None, Sequence[int] | None]) -> None:
        if index < 0:
            idx = len(self._readers) - index
        else:
            idx = index
            
        if idx < 0:
            idx = 0
        elif idx > len(self._readers):
            idx = len(self._readers)
            
        reader, split, series = self.__fix_set_insert_input(value)
        self._readers = self._readers[:idx] + [reader] + self._readers[idx:]
        self._splits = self._splits[:idx] + [split] + self._splits[idx:]
        self._series = self._series[:idx] + [series] + self._series[idx:]

    @property
    def num_series(self) -> int:
        total = 0
        for i, reader in enumerate(self._readers):
            if self._series[i] is None:
                total += len(reader)
            else:
                total += len(self._series[i])
        return total

    def get_series(self, index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if index < 0:
            index = self.num_series + index
            if index < 0:
                raise IndexError("index out of range")
        elif index >= self.num_series:
            raise IndexError("index out of range")
        
        seen = 0
        for i, reader in enumerate(self._readers):
            if self._series[i] is None and index - seen >= len(reader):
                seen += len(reader)
            elif self._series[i] is not None and index - seen >= len(self._series[i]):
                seen += len(self._series[i])
            else:
                if self._series[i] is None:
                    reader_i = i
                    series = reader[index - seen]
                    break
                else:
                    reader_i = i
                    series = reader[self._series[i][index - seen]]
                    break
        else:
            raise IndexError("the time series can't be found")
        
        if self._splits[reader_i] is not None:
            train_pr, _ = self._splits[reader_i]
            last_point = round(series.shape[0] * train_pr)
            return series.iloc[:last_point], series.iloc[last_point:]
        else:
            train_col = rts_config["DEFAULT"]["is_training"]
            if train_col not in series.columns:
                train_pr, _ = self._default_split
                last_point = round(series.shape[0] * train_pr)
                return series.iloc[:last_point], series.iloc[last_point:]
            else:
                return series[series[train_col] == 1], series[series[train_col] == 0]
        
    def get_train_test_split(self, index: int) -> tuple[float, float] | None:
        index = self.__check_index(index)
        return self._splits[index]
    
    def get_series_to_use(self, index: int) -> list[int] | None:
        index = self.__check_index(index)
        series = self._series[index]
        if series is not None:
            return series.copy()
        else:
            return series

    def series_iterator(self, reversed_: bool = False) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        if reversed_:
            indices = reversed(range(self.num_series))
        else:
            indices = range(self.num_series)
        
        for i in indices:
            yield self.get_series(i)

    def set_train_test_split(self,
                             index: int,
                             split: Sequence[float] | None) -> IExperimentLoader:
        index = self.__check_index(index)
        split = self.__fix_input_split(split)
        self._splits[index] = split
        return self

    def set_series_to_use(self,
                          index: int,
                          series: Sequence[int] | None) -> IExperimentLoader:
        index = self.__check_index(index)
        series = self.__fix_input_series(self._readers[index], series)
        self._series[index] = series
        return self
