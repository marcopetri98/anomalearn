from __future__ import annotations

from pathlib import Path
import logging
import os

import numpy as np
import pandas as pd

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class KitsuneReader(IDatasetReader, TSBenchmarkReader):
    """A reader for the Kitsune benchmark.
    
    The reader reads the csv files with the raw data and the labels. The reader
    combines them in the standard format.
    """
    _datasets = ["active_wiretap", "arp_mitm", "fuzzing", "mirai", "os_scan",
                 "ssdp_flood", "ssl_renegotiation", "syn_dos", "video_injection"]
    
    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

        self.__logger = logging.getLogger(__name__)
        self.__check_parameters()
        
    def __len__(self):
        return 9
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("item must be an integer to be able to index")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} datasets")
        
        return self.read(path=item, verbose=False).get_dataframe()
    
    def read(self, path: str | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> KitsuneReader:
        """
        Parameters
        ----------
        path : str or int
            It is the name of the dataset that the method must read or the
            number of the dataset that will be retrieved by the iterator.

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.
        """
        if not isinstance(path, str) and not isinstance(path, int):
            raise TypeError("path must be a string or an int")
        elif not isinstance(path, int) and path not in self._datasets:
            raise ValueError(f"path must be one of {self._datasets}")
        elif not isinstance(path, str) and not 0 <= path < len(self):
            raise ValueError(f"there are only {len(self)} datasets")
        
        dataset_name = path if isinstance(path, str) else self._datasets[path]
        dataset_path = (self._benchmark_path / dataset_name).resolve()

        self.__logger.info(f"reading dataset from {Path(dataset_path) / (dataset_name + '_dataset.csv')}")
        # read raw data and labels
        dataset = pd.read_csv(Path(dataset_path) / (dataset_name + "_dataset.csv"),
                              header=None,
                              dtype=float)
        if dataset_name == "mirai":
            raw_labels = pd.read_csv(Path(dataset_path) / (dataset_name + "_labels.csv"),
                                     header=None,
                                     dtype=int)
            labels = raw_labels[raw_labels.columns[0]].to_numpy()
        else:
            raw_labels = pd.read_csv(Path(dataset_path) / (dataset_name + "_labels.csv"))
            labels = raw_labels["x"].to_numpy()
            
        self.__logger.info("renaming columns with standard names")
        # rename columns with standard names
        dataset_header = [f"{rts_config['Multivariate']['channel_column']}_{e}"
                          for e in dataset.columns]
        columns_mapping = {dataset.columns[i]: dataset_header[i]
                           for i in range(len(dataset_header))}

        # rename columns with standard names
        dataset.rename(columns=columns_mapping, inplace=True)
        
        self.__logger.info("building the final dataframe")
        dataset.set_index(np.arange(dataset.shape[0]), inplace=True)
        dataset.insert(0,
                       rts_config["Multivariate"]["index_column"],
                       np.arange(dataset.shape[0]))
        dataset.insert(len(dataset.columns),
                       rts_config["Multivariate"]["target_column"],
                       labels)
        self._dataset = dataset.copy()
        
        return self
        
    def __check_parameters(self):
        dirs = [e.resolve()
                for e in self._benchmark_path.glob("*")
                if e.is_dir()]
        
        self.__logger.debug(f"dirs contained in the benchmark are {dirs}")
        if len(dirs) != 9:
            raise ValueError("benchmark_location must contain the 9 datasets")
        
        for dir_path in dirs:
            if len(list(dir_path.glob("*"))) != 3:
                raise ValueError("each dataset directory must have exactly 3 "
                                 "files")
