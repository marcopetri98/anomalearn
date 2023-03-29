from . import RangeError


class ClosedOpenRangeError(RangeError):
    """Error raised when a variable does not fall in the closed-open interval.
    """
    def __init__(self, min_, max_, val):
        super().__init__(True, min_, max_, False, val)
