"""
sgd.py - a module for defining the Stochastic Gradient Descent optimizer
"""

import numpy as np

from qoc.models import Optimizer


class SGD(Optimizer):
    """
    a class to define the Stochastic Gradient Descent optimizer
    This implementation follows intuition.
    Fields:
    learning_rate :: float - the initial step size
    """
    name = "sgd"

    def __init__(self, learning_rate=1e-3):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.learning_rate = learning_rate

    def initialize(self, params_shape, params_dtype):
        """Initialize the optimizer for a new optimization series.
        Arguments: none
        Returns: none
        """
        pass


    def update(self, grads, params):
        """Update the learning parameters for the current iteration.
        Args:
        grads :: numpy.ndarray - the gradients of the cost function with
            respect to each learning parameter for the current iteration
        params :: numpy.ndarray - the learning parameters for the current
            iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters for the
            next iteration
        """
        return params - self.learning_rate * grads
        

def _test():
    """
    Run tests on the module.
    """
    sgd = SGD(learning_rate=1)
    params = np.ones(5)
    grads = np.ones(5)
    params = sgd.update(grads, params)
    assert(params.all() == 0)


if __name__ == "__main__":
    _test()
        
