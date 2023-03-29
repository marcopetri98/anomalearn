from . import RangeError


class ClosedRangeError(RangeError):
    """Error raised when a variable does not fall in the closed interval.
    """
    def __init__(self, min_, max_, val):
        super().__init__(True, min_, max_, True, val)
