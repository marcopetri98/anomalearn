from __future__ import annotations

import logging
import os

from .. import IDatasetReader
from . import TSBenchmarkReader, rts_config


class MGABReader(IDatasetReader, TSBenchmarkReader):
    """Data reader for MGAB anomaly benchmark (https://doi.org/10.5281/zenodo.3760086).

    This reader is used to read the datasets contained in the MGAB benchmark.
    """

    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__(benchmark_location=benchmark_location)

        self.__logger = logging.getLogger(__name__)

    def __len__(self):
        return 10

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError("use __getitem__ only to iterate over time series")
        elif not 0 <= item < len(self):
            raise IndexError(f"there are only {len(self)} series in the dataset")

        return self.read(path=item, verbose=False).get_dataframe()

    def read(self, path: str | bytes | os.PathLike | int,
             file_format: str = "csv",
             pandas_args: dict | None = None,
             *args,
             **kwargs) -> MGABReader:
        """
        Parameters
        ----------
        path : str or bytes or PathLike or int
            The path to the csv containing the time series or the integer
            representing which time series to load from the dataset location.

        file_format : str, default="csv"
            Ignored.

        pandas_args : dict or None, default=None
            Ignored.

        Returns
        -------
        self : MGABReader
            An instance to itself to allow call chaining.
        """
        if not isinstance(path, int) and not os.path.isfile(path):
            raise ValueError("path must be a valid path or an int")
        elif isinstance(path, int) and not 0 <= path < len(self):
            raise ValueError(f"path must be between 0 and {len(self)}")

        if isinstance(path, int):
            path = self._benchmark_path / (str(path + 1) + ".csv")

        super().read(path=path,
                     file_format=file_format,
                     pandas_args=pandas_args,
                     verbose=False)

        self.__logger.info("renaming columns with standard names")
        self._dataset.rename(columns={
                                "Unnamed: 0": rts_config["Univariate"]["index_column"],
                                "value": rts_config["Univariate"]["value_column"],
                                "is_anomaly": rts_config["Univariate"]["target_column"]
                            },
                            inplace=True)

        return self
