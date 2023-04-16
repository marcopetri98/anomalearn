from __future__ import annotations

from pathlib import Path
import logging
import os

import numpy as np
import pandas as pd

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class ExathlonReader(IDatasetReader, TSBenchmarkReader):
    """A reader for the Exathlon benchmark (https://doi.org/10.14778/3476249.3476307).

    The reader automatically provides an easy-to-use API to read the normal and
    abnormal series (undisturbed and disturbed traces respectively).
    """
    _all_modes = ["train", "test", "all"]

    def __init__(self, benchmark_location: str | os.PathLike,
                 mode: str = "all"):
        super().__init__(benchmark_location=benchmark_location)

        self.__logger = logging.getLogger(__name__)
        self._mode = mode

        self.__check_parameters()

        def _exathlon_file_order(el):
            first_num = int(el.split("_")[0]) * 1e10
            second_num = int(el.split("_")[1]) * 1e8
            third_num = int(el.split("_")[2])
            fourth_num = int(el.split("_")[3])
            return first_num + second_num + third_num + fourth_num

        self._gt = pd.read_csv(self._benchmark_path / "ground_truth.csv")
        self._disturbed = self._gt["trace_name"].unique().tolist()
        self._disturbed.sort(key=_exathlon_file_order)

        self._files_paths = []
        for root, _, files in os.walk(self._benchmark_location):
            for name in files:
                if name != "ground_truth.csv":
                    self._files_paths.append(os.path.normpath(os.path.join(root, name)))
        self._files_paths.sort(key=lambda el: int(el.split(os.sep)[-2].split("app")[1]) * 1e12 + _exathlon_file_order(el.split(os.sep)[-1].split(".")[0]))

        self._disturbed_paths = [e
                                 for e in self._files_paths
                                 if os.path.basename(e).split(".")[0] in self._disturbed]
        self._undisturbed_paths = [e
                                   for e in self._files_paths
                                   if e not in self._disturbed_paths]

    def set_mode(self, mode: str) -> None:
        """Sets the mode for the reader.

        Parameters
        ----------
        mode : ["all", "train", "test"]
            Sets the mode in which the reader must be used.

        Returns
        -------
        None
        """
        self._mode = mode
        self.__check_parameters()

    def __len__(self):
        match self._mode:
            case "all":
                return len(self._files_paths)

            case "train":
                return len(self._files_paths) - len(self._disturbed)

            case "test":
                return len(self._disturbed)

            case _:
                raise ValueError("mode has an invalid value")

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} time series to read")

        return self.read(path=item, verbose=False).get_dataframe()

    def read(self, path: int | str,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             full_rename: bool = True,
             *args,
             **kwargs) -> ExathlonReader:
        """
        Parameters
        ----------
        path : int or str
            It is either the index of the time series to read or the name of the
            time series that the user wants to read without the file extension
            (e.g. "1_0_10000_17" is a valid name).

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.

        full_rename : bool, default=True
            If `full_rename` is `True` the channels are renamed with integers
            going from 0 to N. Differently, the standard names are kept and the
            standard names are only prepended.
        """
        if not isinstance(path, int) and not isinstance(path, str):
            raise TypeError("path must be either int or string")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise ValueError(f"there are {len(self)} series with mode {self._mode}")
        elif not isinstance(full_rename, bool):
            raise TypeError("full_rename must be boolean")
        elif isinstance(path, str) and not any(path in e for e in self._files_paths):
            raise ValueError("path must be a valid name of a series")

        # load the time series selected by the user
        if isinstance(path, str):
            path_idx = [path in e for e in self._files_paths].index(True)
            path = self._files_paths[path_idx]
        else:
            match self._mode:
                case "all":
                    if path < len(self._undisturbed_paths):
                        path = self._undisturbed_paths[path]
                    else:
                        path = self._disturbed_paths[path - len(self._undisturbed_paths)]

                case "train":
                    path = self._undisturbed_paths[path]

                case "test":
                    path = self._disturbed_paths[path]

        self.__logger.info(f"reading time series at {path}")
        # actually load the csv using pandas
        dataset = pd.read_csv(path)

        # build the target vector
        trace_name = Path(path).name.split(".")[0]
        target = np.zeros(dataset.shape[0])
        if path in self._disturbed_paths:
            gt: pd.DataFrame = self._gt.loc[self._gt["trace_name"] == trace_name]
            for _, row in gt.iterrows():
                if np.isnan(row["extended_effect_end"]):
                    end = int(row["root_cause_end"])
                else:
                    end = int(row["extended_effect_end"])
                start = int(row["root_cause_start"])

                if start not in dataset["t"].tolist():
                    self.__logger.warning("reading dataset whose start is not "
                                          "present. The next timestamp will be "
                                          "selected")
                    start += 1
                if end not in dataset["t"].tolist():
                    self.__logger.warning("reading dataset whose end is not "
                                          "present. The previous timestamp will "
                                          "be selected")
                    end -= 1
                
                start_idx = dataset["t"].tolist().index(start)
                end_idx = dataset["t"].tolist().index(end)
                target[start_idx:end_idx + 1] = 1

        self.__logger.info("renaming columns with standard names")
        # build columns name mappings
        channels = {e: f"channel_{e if not full_rename else idx}"
                    for idx, e in enumerate(dataset.columns[1:])}

        # rename columns
        dataset.rename(columns={"t": rts_config["Multivariate"]["index_column"]},
                       inplace=True)
        dataset.rename(columns=channels, inplace=True)
        dataset.insert(len(dataset.columns),
                       rts_config["Multivariate"]["target_column"],
                       target)

        self._dataset = dataset.copy()

        return self

    def __check_parameters(self):
        allowed_content = {"app1", "app2", "app3", "app4", "app5", "app6",
                           "app7", "app8", "app9", "app10", "ground_truth.csv"}
        contents = [e.name for e in self._benchmark_path.glob("*")]
        self.__logger.debug(f"benchmark contents are {contents}")

        if not isinstance(self._mode, str):
            raise TypeError(f"mode must be one of {self._all_modes}")

        if len(contents) < 11:
            raise ValueError("benchmark_location must contain the 10 apps' "
                             "folders with data and the ground truth")
        elif len(allowed_content.difference(allowed_content.intersection(contents))) != 0:
            raise ValueError("benchmark_location must contain the following "
                             f"folders and libraries {allowed_content}")
