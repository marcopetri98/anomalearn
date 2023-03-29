

class SelectionError(ValueError):
    """Error raised when a variable does not take a value from a fixed list.
    """
    def __init__(self, values, val, message: str = None):
        self.values = values
        self.val = val
        
        if message is None:
            self.message = f"Expected one of {values}, got {val}"
        else:
            self.message = message
        
        super().__init__(self.message)
