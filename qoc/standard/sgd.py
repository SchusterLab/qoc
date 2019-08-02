"""
sgd.py - a module for defining the Stochastic Gradient Descent optimizer
"""

import numpy as np

from qoc.models import Optimizer
from qoc.util import (complex_to_real_imag_vec,
                      real_imag_to_complex_vec)

class SGD(Optimizer):
    """
    a class to define the Stochastic Gradient Descent optimizer
    This implementation follows intuition.
    Fields:
    complex_params :: bool - whether or not the parameters are complex
    learning_rate :: float - the initial step size
    """
    name = "sgd"

    def __init__(self, learning_rate=1e-3):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.complex_params = False
        self.learning_rate = learning_rate


    def initialize(self, params_shape, params_dtype):
        """Initialize the optimizer for a new optimization series.
        Arguments:
        params_shape :: tuple(int) - the shape of the learning parameters
        Returns: none
        """
        if params_dtype in (np.complex64, np.complex128):
            self.complex_params = True
            params_shape = (2, *params_shape)


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
        if self.complex_params:
            grads = complex_to_real_imag_vec(grads)
            params = complex_to_real_imag_vec(params)

        params = params - self.learning_rate * grads

        if self.complex_params:
            params = real_imag_to_complex_vec(params)

        return params
        

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
        
