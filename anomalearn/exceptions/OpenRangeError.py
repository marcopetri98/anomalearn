from . import RangeError


class OpenRangeError(RangeError):
    """Error raised when a variable does not fall in the open interval.
    """
    def __init__(self, min_, max_, val):
        super().__init__(False, min_, max_, False, val)
