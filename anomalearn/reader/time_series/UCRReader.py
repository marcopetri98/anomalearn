from __future__ import annotations

from pathlib import Path
import logging
import os

import numpy as np
import pandas as pd

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class UCRReader(IDatasetReader, TSBenchmarkReader):
    """Data reader for UCR benchmark (https://doi.org/10.1109/TKDE.2021.3112126).

    The reader loads the txt files from the UCR benchmark and translate them in
    the default format.
    """

    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

        self._all_datasets = [e.resolve()
                              for e in self._benchmark_path.glob("*")
                              if e.is_file()]
        self._all_datasets = sorted(self._all_datasets)
        self.__logger = logging.getLogger(__name__)

        self.__check_parameters()

    def __len__(self):
        return 250

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("the index must be an int")

        return self.read(item, verbose=False).get_dataframe()

    def read(self, path: str | bytes | os.PathLike | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> UCRReader:
        """
        Parameters
        ----------
        path : str or bytes or PathLike or int
            It is a path to the txt file containing the time series formatted
            as it is formatted in the benchmark or an integer representing the
            time series to load from the benchmark (indexed from 0).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.
        """
        if not os.path.isfile(path) and not isinstance(path, int):
            raise ValueError("path must be a path to a file or an int")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise IndexError(f"there are only {len(self)} series")

        if isinstance(path, int):
            path = (self._benchmark_path / self._all_datasets[path]).resolve()

        self.__logger.info(f"reading dataset from {str(path)}")
        data = []
        with open(path, "r", encoding="utf-8") as f:
            contents = f.read()
            is_multiline = contents.count('\n') != 1

            if is_multiline:
                numbers = contents.split("\n")
            else:
                numbers = contents.split(" ")
            numbers = [e.strip() for e in numbers if len(e.strip()) != 0]

            for number in numbers:
                data.append(float(number))
        data = np.array(data)

        self.__logger.info(f"reading anomalies for {str(path)}")
        filename = Path(path).name
        filename_parts = filename.split("_")
        last_train_point = int(filename_parts[-3])
        anomaly_start = int(filename_parts[-2])
        anomaly_end = int(filename_parts[-1].split(".")[0])

        self.__logger.info(f"creating an index for {str(path)}")
        labels = np.zeros(data.shape)
        labels[anomaly_start:anomaly_end + 1] = 1
        timestamps = np.arange(data.shape[0])
        is_training = np.zeros(data.shape)
        is_training[:last_train_point + 1] = 1
        all_data = np.concatenate((timestamps.reshape(-1, 1),
                                   data.reshape(-1, 1),
                                   labels.reshape(-1, 1),
                                   is_training.reshape(-1, 1)),
                                  axis=1)

        self._dataset = pd.DataFrame(all_data,
                                     columns=[rts_config["Univariate"]["index_column"],
                                              rts_config["Univariate"]["value_column"],
                                              rts_config["Univariate"]["target_column"],
                                              rts_config["Univariate"]["is_training"]])

        return self

    def __check_parameters(self):
        if len(self._all_datasets) != len(self):
            self.__logger.debug(f"benchmark folder contains {self._all_datasets}")
            raise ValueError(f"The benchmark should contain {len(self)} files")
