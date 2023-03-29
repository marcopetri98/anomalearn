from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .. import TSReader, rts_config
from ... import IDatasetReader


class NASAReader(IDatasetReader, TSReader):
    """Data reader for NASA MSL and NASA SMAP datasets (https://doi.org/10.1145/3219819.3219845).

    The reader is set up such that NASA files pre-split in training and testing
    are read as is. Eventually, with a flag is possible to decide to merge
    training and testing data to build a single unique dataframe containing all
    data such that new and different split can be performed.
    """
    def __init__(self, anomalies_path: str | os.PathLike):
        super().__init__()

        self.__logger = logging.getLogger(__name__)
        self._anomalies_path = Path(anomalies_path)
        self._channels = list(sorted([e.name.split(".")[0]
                                      for e in (Path(self._anomalies_path).parent / "train").glob("*")]))

        self.__check_parameters()

        self._anomalies_df = pd.read_csv(self._anomalies_path)

    def __len__(self):
        return 82

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("the index must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} channels in total")

        return self.read(path=item, merge_split=True, verbose=False).get_dataframe()

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             dataset_folder: str = "same-as-labels",
             *args,
             **kwargs) -> NASAReader:
        """
        Parameters
        ----------
        path : str or int
            The names of the channels to read (e.g., "A-1" is a valid value
            for path), or an integer stating which time series to load from
            the benchmark (indexed from 0).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.

        dataset_folder : str, default="same-as-labels"
            It is the path of the folder containing training and testing splits
            of the dataset. Otherwise, the option "same-as-labels" assumes that
            the dataset folder is the same folder containing the labels.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an integer")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise IndexError(f"path is {path} and NASA has {len(self)} series")
        elif dataset_folder != "same-as-labels" and not os.path.isdir(dataset_folder):
            raise TypeError("dataset_folder must be a valid path to a dir")

        if isinstance(path, int):
            path = self._channels[path]

        if dataset_folder == "same-as-labels":
            dataset_folder = Path(self._anomalies_path).parent

        if path not in self._channels:
            raise ValueError("path must be a valid channel name")
        elif not {"train", "test"}.issubset(os.listdir(dataset_folder)):
            raise ValueError("train and test folders are not present, pass a "
                             "valid dataset folder")

        row_selector = self._anomalies_df["chan_id"] == path

        # if the user specified one of the channels build the path
        train_path = Path(dataset_folder) / "train" / (path + ".npy")
        test_path = Path(dataset_folder) / "test" / (path + ".npy")
        self.__logger.debug(f"train={str(train_path)}, test={str(test_path)}")

        self.__logger.info(f"reading training dataset at {str(train_path)}")
        self.__logger.info(f"reading testing dataset at {str(test_path)}")
        train_series = np.load(str(train_path))
        test_series = np.load(str(test_path))

        train_labels = np.zeros(train_series.shape[0])
        test_labels = np.zeros(test_series.shape[0])
        anomalies = self._anomalies_df.loc[row_selector]["anomaly_sequences"]
        for i in range(anomalies.shape[0]):
            interval = ast.literal_eval(anomalies.iloc[i])
            for sequence in interval:
                test_labels[sequence[0]:sequence[1] + 1] = 1

        self.__logger.info("renaming columns with standard names")
        columns = [str(e) for e in range(train_series.shape[1])]
        columns[0] = "telemetry"
        columns = [rts_config["Multivariate"]["channel_column"] + "_" + e
                   for e in columns]

        self.__logger.info("finishing the dataframe")
        series = np.concatenate((train_series, test_series))
        targets = np.concatenate((train_labels, test_labels))
        timestamp = np.arange(series.shape[0])
        is_training = np.zeros(series.shape[0])
        is_training[:train_series.shape[0]] = 1
        all_columns = [rts_config["Multivariate"]["index_column"]]
        all_columns.extend(columns)
        all_columns.append(rts_config["Multivariate"]["target_column"])
        all_columns.append(rts_config["Multivariate"]["is_training"])
        all_data = np.concatenate((timestamp.reshape(-1, 1),
                                   series,
                                   targets.reshape(-1, 1),
                                   is_training.reshape(-1, 1)),
                                  axis=1)
        self._dataset = pd.DataFrame(all_data, columns=all_columns)

        return self

    def __check_parameters(self):
        self.__logger.debug(f"path to anomalies is {str(self._anomalies_path)}")

        if not self._anomalies_path.is_file():
            raise ValueError("anomalies_path must be a path to a csv file")
