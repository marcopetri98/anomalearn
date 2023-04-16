from __future__ import annotations

import logging
import os

import pandas as pd

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class GHLReader(IDatasetReader, TSBenchmarkReader):
    """Data reader for GHL dataset (https://doi.org/10.48550/arXiv.1612.06676).

    The reader is able to read both testing and training dataset with meaningful
    interfaces.
    """
    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

        self.__logger = logging.getLogger(__name__)
        self.__check_parameters()

        self._all_test_sets_paths = list(sorted([str(e.resolve()) for e in self._benchmark_path.glob("[0-9][0-9]*.csv")]))
        self._train_set_path = str(list(self._benchmark_path.glob("train*.csv"))[0])

    def __len__(self):
        return 49

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} testing sets")

        if item == 0:
            return self.read(path="train", verbose=False).get_dataframe()
        else:
            return self.read(path=item - 1, verbose=False).get_dataframe()

    def read(self, path: int | str,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             full_rename: bool = True,
             *args,
             **kwargs) -> GHLReader:
        """
        Parameters
        ----------
        path : int or "train"
            It is the number of the testing set that must be read or a string
            equal to "train" to retrieve the training set.

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.

        full_rename : bool, default=True
            If `full_rename` is `True` the channels are renamed with integers
            going from 0 to N. Differently, the dataset names are kept and the
            standard names are only prepended.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be an integer or a string")
        elif isinstance(path, str) and path != "train":
            raise TypeError("path can only be \"train\" if it is a string")
        elif not isinstance(full_rename, bool):
            raise TypeError("full_rename must be boolean")
        elif isinstance(path, int) and not 0 <= path < len(self) - 1:
            raise ValueError(f"there are only {len(self)} testing sets")

        if isinstance(path, int):
            file_path = self._all_test_sets_paths[path]
        else:
            file_path = self._train_set_path

        if path == "train":
            self.__logger.info("reading training")
        else:
            self.__logger.info("reading testing")

        self.__logger.info(f"reading time series at {file_path}")
        # read file and reorder columns
        dataset = pd.read_csv(file_path)
        ordered_cols = [e for e in dataset.columns if e not in ["DANGER", "FAULT", "ATTACK"]]
        if path != "train":
            ordered_cols.extend(["DANGER", "FAULT", "ATTACK"])
        dataset = dataset[ordered_cols]

        self.__logger.info("renaming columns with standard names")
        # build the list of columns without classes maintaining the order
        cols = dataset.columns[1:].tolist()
        for class_col in ["DANGER", "FAULT", "ATTACK"]:
            if class_col in cols:
                cols.remove(class_col)
        # build columns name mappings
        channels = {e: f"channel_{e if not full_rename else idx}"
                    for idx, e in enumerate(cols)}
        classes = {e: f"class_{e if not full_rename else idx}"
                   for idx, e in enumerate(["DANGER", "FAULT", "ATTACK"])}
        
        self.__logger.info(f"the renaming of the columns is {channels}")
        self.__logger.info(f"the renaming of the classes is {classes}")

        # rename columns
        dataset.rename(columns={"Time": rts_config["Multivariate"]["index_column"]},
                       inplace=True)
        dataset.rename(columns=channels, inplace=True)
        dataset.rename(columns=classes, inplace=True)

        self._dataset = dataset.copy()

        return self

    def __check_parameters(self):
        benchmark_contents = list(self._benchmark_path.glob("*"))
        self.__logger.debug(f"benchmark folder contains {benchmark_contents}")
        if len(benchmark_contents) != len(self):
            raise ValueError("benchmark_location must contain all the 48 tests "
                             "and the training set")
