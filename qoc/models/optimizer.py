"""
optimizer.py - a class to define a class to encapsulate gradient based
optimizers
"""

class Optimizer(object):
    """
    a class to encapsulate a gradient based optimizer
    Fields: none
    Methods:
    init :: (*) -> None - see method definition
    update :: (params :: numpy.ndarray, grads :: numpy.ndarray)
              -> new_params :: numpy.ndarray - see method definition
    """

    def __init__(self):
        super().__init__()
    

    def initialize(self):
        """Initialize the optimizer for a new optimization series.
        Args: none
        Returns: none
        """
        pass


    def update(self, params, grads):
        """Update the learning parameters for the current iteration.
        Args:
        params :: numpy.ndarray - the learning parameters for the current
                                  iteration
        grads :: numpy.ndarray - the gradients of the cost function with respect
                                 to each learning parameter for the current
                                 iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters to be used for the
                                      next iteration
        """
        return params
