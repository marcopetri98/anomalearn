

class RangeError(ValueError):
    """Error raised when a variable does not fall in the interval.
    """
    def __init__(self, left, min_, max_, right, val, message: str = None):
        self.left = left
        self.min = min_
        self.max = max_
        self.right = right
        self.val = val
        
        par_left = "[" if self.left else "("
        par_right = "]" if self.right else ")"

        if message is None:
            self.message = f"Expected value in range {par_left}{min_},{max_}{par_right}, got {val}"
        else:
            self.message = message
        
        super().__init__(self.message)
