

class NotTrainedError(ValueError):
    """Error raised when a processing operation has been called without fitting the estimator.
    """
    def __init__(self, message: str = None):
        if message is None:
            self.message = "The estimator has not been fit. Fit it before calling this function"
        else:
            self.message = message
        
        super().__init__(self.message)
