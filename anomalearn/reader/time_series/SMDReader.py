from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class SMDReader(IDatasetReader, TSBenchmarkReader):
    """Data reader for SMD dataset (https://doi.org/10.1145/3292500.3330672).

    The reader reads the txt files in the SMD benchmark folder and translates
    them into the default format for time series.
    """
    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

        self.__logger = logging.getLogger(__name__)
        self._interpretation = self._benchmark_path / "interpretation_label"
        self._test_set = self._benchmark_path / "test"
        self._test_gt = self._benchmark_path / "test_label"
        self._train_set = self._benchmark_path / "train"
        
        self._machines = [e.name.split(".")[0]
                          for e in self._train_set.glob("*.txt")
                          if e.is_file()]
        self._machines.sort(key=lambda elem: int(elem.split("-")[1]) * 10 + int(elem.split("-")[2]))

        self.__check_parameters()

    def __len__(self):
        return 28

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("the index must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} machines")

        return self.read(path=item, verbose=False).get_dataframe()

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> SMDReader:
        """
        Parameters
        ----------
        path : str or int
            It is the name of the machine that you want to read (e.g.,
            machine-1-1"), or an integer stating which time series to load from
            the benchmark (indexed from 0).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a machine name or an index")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise IndexError(f"path is {path} and SMD has {len(self)} series")
        elif isinstance(path, str) and path not in self._machines:
            raise ValueError(f"path must be one of {self._machines}")

        if isinstance(path, int):
            path = self._machines[path]

        self.__logger.info(f"reading training file at {str(self._train_set / (path + '.txt'))}")
        self.__logger.info(f"reading testing file at {str(self._test_set / (path + '.txt'))}")
        self.__logger.info(f"reading ground truth file at {str(self._test_gt / (path + '.txt'))}")
        # read training dataset and testing
        training_set = pd.read_csv(self._train_set / (path + ".txt"),
                                   header=None)
        testing_set = pd.read_csv(self._test_set / (path + ".txt"),
                                  header=None)
        test_labels = pd.read_csv(self._test_gt / (path + ".txt"),
                                  header=None)[0].to_numpy()

        self.__logger.info("renaming columns with standard names")
        # retrieve number of columns and mapping to new columns' names
        dataset_header = [f"{rts_config['Multivariate']['channel_column']}_{e}"
                          for e in training_set.columns]
        columns_mapping = {training_set.columns[i]: dataset_header[i]
                           for i in range(len(dataset_header))}

        # rename columns with standard names
        training_set.rename(columns=columns_mapping, inplace=True)
        testing_set.rename(columns=columns_mapping, inplace=True)

        self.__logger.info(f"building {rts_config['Multivariate']['target_column']}"
                           f" and {rts_config['Multivariate']['is_training']} "
                           f"columns")
        # build overall labels and training column
        labels = np.zeros(training_set.shape[0] + testing_set.shape[0])
        labels[training_set.shape[0]:] = test_labels
        is_training = np.zeros(training_set.shape[0] + testing_set.shape[0])
        is_training[:training_set.shape[0]] = 1
        interpretation = [None] * labels.shape[0]

        self.__logger.info("extracting anomalies interpretation")
        # reading the interpretation file
        with open(self._interpretation / (path + ".txt"), "r", encoding="utf-8") as f:
            for line in f:
                interval, channels = line.split(":")
                start, end = interval.split("-")
                elements = channels.split("\n")[0].split(",")
                elements = [int(e) for e in elements]
                for i in range(int(start), int(end) + 1, 1):
                    interpretation[i] = elements

        self.__logger.info("building final dataframe")
        # build the overall dataset
        self._dataset = pd.concat((training_set, testing_set))
        self._dataset.set_index(np.arange(self._dataset.shape[0]), inplace=True)
        self._dataset.insert(0,
                             rts_config["Multivariate"]["index_column"],
                             np.arange(self._dataset.shape[0]))
        self._dataset.insert(len(self._dataset.columns),
                             rts_config["Multivariate"]["target_column"],
                             labels)
        self._dataset.insert(len(self._dataset.columns),
                             rts_config["Multivariate"]["is_training"],
                             is_training)
        self._dataset.insert(len(self._dataset.columns),
                             "interpretation",
                             interpretation)

        return self

    def __check_parameters(self):
        self.__logger.debug(f"benchmark_location contents are "
                            f"{list(self._benchmark_path.glob('*'))}")
        
        if not self._interpretation.is_dir():
            raise ValueError("benchmark_location must contain a folder named "
                             "interpretation_label")
        elif not self._test_gt.is_dir():
            raise ValueError("benchmark_location must contain a folder named "
                             "test_label")
        elif not self._test_set.is_dir():
            raise ValueError("benchmark_location must contain a folder named "
                             "test")
        elif not self._train_set.is_dir():
            raise ValueError("benchmark_location must contain a folder named "
                             "train")
