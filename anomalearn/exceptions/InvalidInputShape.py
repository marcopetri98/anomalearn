

class InvalidInputShape(Exception):
    """An exception thrown if the input array has invalid shape.
    """
    def __init__(self, expected_shape: tuple, shape: tuple, message: str = None):
        self.expected_shape = expected_shape
        self.shape = shape
        
        if message is None:
            self.message = f"Received shape {expected_shape} with expected shape {shape}"
        else:
            self.message = message
        
        super().__init__(self.message)
