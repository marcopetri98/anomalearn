import abc
from abc import ABC


class ITimeSeriesAnomalyWrapper(ABC):
    """Interface identifying a time series anomaly detector wrapping another approach.
    """
    
    @abc.abstractmethod
    def _build_wrapped(self) -> None:
        """Instantiates the wrapped model.

        Returns
        -------
        None
        """
        pass
