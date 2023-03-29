import os
from pathlib import Path

from . import TSReader


class TSBenchmarkReader(TSReader):
    """A time series benchmark reader.

    Parameters
    ----------
    benchmark_location : str
        The location of the benchmark's folder.
    """

    def __init__(self, benchmark_location: str | os.PathLike):
        super().__init__()

        self._benchmark_location = benchmark_location
        self._benchmark_path = Path(self._benchmark_location)

        self.__check_parameters()

    def __check_parameters(self):
        """Check parameters.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any parameter has wrong type.

        ValueError
            If any parameter has wrong value.
        """
        if not isinstance(self._benchmark_location, str) and not isinstance(self._benchmark_location, os.PathLike):
            raise TypeError("benchmark_location must be a string or a path-like")

        path = Path(self._benchmark_location)

        if not path.is_dir():
            raise ValueError(f"benchmark_location must be a directory (it is {str(path)})")
