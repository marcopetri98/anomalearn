from . import IDataReader
from .. import ObtainableABC


class IDatasetReader(IDataReader, ObtainableABC):
    """Abstract class for all dataset readers.
    
    A dataset reader should implement this class if it is to be considered a
    data reader.
    """
    pass
