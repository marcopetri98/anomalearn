from . import RangeError


class OpenClosedRangeError(RangeError):
    """Error raised when a variable does not fall in the open-closed interval.
    """
    def __init__(self, min_, max_, val):
        super().__init__(False, min_, max_, True, val)
