"""
optimizer.py - a module to define a class to encapsulate gradient based
optimizers
"""

class Optimizer(object):
    """
    a class to encapsulate a gradient based optimizer
    Fields:
    name :: str - the identifier for the optimizer
    """
    name = "parent_optimizer"

    def __init__(self):
        super().__init__()


    def __str__(self):
        return self.name


    def __repr__(self):
        return self.__str__()
    

    def initialize(self, params_shape):
        """Initialize the optimizer for a new optimization series.
        Args:
        params_shape :: tuple(int) - the shape of the learning parameters
        Returns: none
        """
        pass


    def update(self, grads, params):
        """Update the learning parameters for the current iteration.
        Args:
        grads :: numpy.ndarray - the gradients of the cost function with respect
                                 to each learning parameter for the current
                                 iteration
        params :: numpy.ndarray - the learning parameters for the current
                                  iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters to be used for the
                                      next iteration
        """
        pass
