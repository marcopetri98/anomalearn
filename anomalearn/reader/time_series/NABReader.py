from __future__ import annotations

import json
import logging
import operator
import os

import numpy as np

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class NABReader(IDatasetReader, TSBenchmarkReader):
    """A reader of NAB time series datasets.

    The reader reads the time series and adds the target column class for the
    time series defined as it is defined by NAB (windows such that the sum of
    the windows' length is 10% of data around each label point).
    """
    
    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

        self.__logger = logging.getLogger(__name__)
        self._datasets_paths = []
        self._datasets_names = []

        data_path = self._benchmark_path / "data"
        for path in data_path.glob("**/*"):
            if path.is_file() and path.name != "README.md":
                self._datasets_names.append(path.name.split(".")[0])
                self._datasets_paths.append(path.resolve())
                    
        # order paths and names as in path
        sorted_couples = sorted(list(zip(self._datasets_paths, self._datasets_names)), key=operator.itemgetter(0))
        self._datasets_paths = [e[0] for e in sorted_couples]
        self._datasets_names = [e[1] for e in sorted_couples]

        labels_path = self._benchmark_path / "labels"
        windows_path = (labels_path / "combined_windows.json").resolve()
        with open(windows_path, encoding="utf-8") as file:
            self._combined_windows = json.load(file)

        self._combined_windows = {key.split("/")[1].split(".")[0]: self._combined_windows[key]
                                  for key in self._combined_windows}
        
        self.__check_parameters()

    def __len__(self):
        return len(self._datasets_names)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"item must be less than {len(self)}")

        return self.read(path=item, verbose=False).get_dataframe()

    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> NABReader:
        """
        Parameters
        ----------
        path : str
            It is the name of the dataset that you want to read (e.g.,
            art_daily_no_noise), or an integer stating which time series to load
            from the benchmark (indexed from 0).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an int")
        elif isinstance(path, str) and path not in self._datasets_names:
            raise ValueError(f"path must be one of {self._datasets_names}")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise ValueError(f"there are only {len(self)} datasets in NAB")
        
        # get the dataset path
        if isinstance(path, str):
            dataset_path = self._datasets_paths[self._datasets_names.index(path)]
            dataset_name = path
        else:
            dataset_path = self._datasets_paths[path]
            dataset_name = self._datasets_names[path]

        # load dataset
        super().read(dataset_path, "csv", pandas_args)

        self.__logger.info("building point labels from combined windows")
        # build target class vector
        target = np.zeros(self._dataset.shape[0])
        if len(self._combined_windows[dataset_name]) != 0:
            for window in self._combined_windows[dataset_name]:
                start_idx = self._dataset["timestamp"].tolist().index(window[0].split(".")[0])
                end_idx = self._dataset["timestamp"].tolist().index(window[1].split(".")[0])
                target[start_idx:end_idx + 1] = 1

        self.__logger.info("renaming columns with standard names")
        # give to the columns standard names
        self._dataset.rename(columns={"timestamp": rts_config["Univariate"]["index_column"]},
                             inplace=True)
        self._dataset.rename(columns={"value": rts_config["Univariate"]["value_column"]},
                             inplace=True)
        self._dataset.insert(len(self._dataset.columns),
                             rts_config["Univariate"]["target_column"],
                             target)
        
        return self

    def __check_parameters(self):
        if len(list(self._benchmark_path.glob("*"))) != 2:
            raise ValueError("benchmark_location must contain only data and "
                             "labels folders")

        labels_path = self._benchmark_path / "labels"
        data_path = self._benchmark_path / "data"
        
        self.__logger.debug(f"labels path contains {list(labels_path.glob('*'))}")
        if "combined_windows.json" not in [str(e.name) for e in labels_path.glob("*")]:
            raise ValueError("labels folder does not contain combined_windows "
                             "file")

        num_dirs = 0
        num_files = 0
        for _, dirs, files in os.walk(data_path):
            num_dirs += len(dirs)
            num_files += len(files)

        if num_dirs != 7 or num_files != len(self) + 1:
            raise ValueError("data folder should contain the 7 NAB folders and "
                             f"all the {len(self)} files and the readme")
