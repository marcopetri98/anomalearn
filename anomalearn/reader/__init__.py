"""
Reading module: support for datasets reading.

This module is intended to support datasets reading from file. The reading is
performed using python libraries as pandas or the standard python reading
libraries. This module has the unique aim of easing the reading of datasets by
employing always the same interface for datasets reading and hiding in the call
the eventual complexity of parsing multiple files and combining files into a
unique dataframe representing the dataset. However, this module does not make
any distinction between the types of datasets that can be read from it in the
interfaces. Classes may distinguish between class of problems such as image
datasets or time series. However, no assumption is performed on the task that
will be carried on with the dataset. Therefore, a general time series reader
will be simply usable for all the time series.
"""

from .IDataMultipleReader import IDataMultipleReader
from .IDataReader import IDataReader
from .IDatasetReader import IDatasetReader
from .MissingStrategy import MissingStrategy
