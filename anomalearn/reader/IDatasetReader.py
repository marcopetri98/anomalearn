from .. import ObtainableABC
from . import IDataReader


class IDatasetReader(IDataReader, ObtainableABC):
    """Abstract class for all dataset readers.
    
    A dataset reader should implement this class if it is to be considered a
    data reader.
    """
