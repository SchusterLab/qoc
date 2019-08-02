"""
adam.py - a module for defining the Adam optimizer
"""

import numpy as np

from qoc.models import Optimizer
from qoc.util import (complex_to_real_imag_vec,
                      real_imag_to_complex_vec)

class Adam(Optimizer):
    """
    a class to define the Adam optimizer
    This implementation follows the original algorithm
    https://arxiv.org/abs/1412.6980.
    Fields:
    beta_1 :: float - gradient decay bias
    beta_2 :: float - gradient squared decay bias
    complex_params :: bool - whether or not the parameters require
        a transformation on updating
    epsilon :: float - fuzz factor
    gradient_moment :: numpy.ndarray - running optimization variable
    gradient_square_moment :: numpy.ndarray - running optimization variable
    iteration_count :: int - the current count of iterations performed
    learning_rate :: float - the initial step size
    name :: str - identifier for the optimizer
    """
    name = "adam"

    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 learning_rate=1e-3,):
        """
        See class definition for argument specifications.
        Default values are chosen in accordance to those proposed
        in the paper.
        """
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.complex_params = False
        self.epsilon = epsilon
        self.gradient_moment = None
        self.gradient_square_moment = None
        self.learning_rate = learning_rate


    def __str__(self):
        return ("{}, beta_1: {}, beta_2: {}, epsilon: {}, lr: {}"
                "".format(self.name, self.beta_1, self.beta_2,
                          self.epsilon, self.learning_rate,))

    
    def initialize(self, params_shape, params_dtype):
        """Initialize the optimizer for a new optimization series.
        Arguments:
        params_shape :: tuple(int) - the shape of the learning parameters
        Returns: none
        """
        # Double the shape of the parameters if they are complex.
        # A C -> R2 transformation will be applied.
        if params_dtype in (np.complex64, np.complex128):
            self.complex_params = True
            params_shape = (2, *params_shape)
        self.iteration_count = 0
        self.gradient_moment = np.zeros(params_shape)
        self.gradient_square_moment = np.zeros(params_shape)

        
    def update(self, grads, params):
        """Update the learning parameters for the current iteration.
        Args:
        grads :: numpy.ndarray - the gradients of the cost function with
            respect to each learning parameter for the current iteration
        params :: numpy.ndarray - the learning parameters for the
            current iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters to be used
            for the next iteration
        """
        if self.complex_params:
            grads = complex_to_real_imag_vec(grads)
            params = complex_to_real_imag_vec(params)
        
        self.iteration_count += 1
        self.gradient_moment = (self.beta_1 * self.gradient_moment
                                + (1 - self.beta_1) * grads)
        self.gradient_square_moment = (self.beta_2 * self.gradient_square_moment
                                       + (1 - self.beta_2) * np.square(grads))
        gradient_moment_hat = np.divide(self.gradient_moment,
                                        1 - np.power(self.beta_1, self.iteration_count))
        gradient_square_moment_hat = np.divide(self.gradient_square_moment,
                                               1 - np.power(self.beta_2, self.iteration_count))
        params = params - self.learning_rate * np.divide(gradient_moment_hat,
                                                       np.sqrt(gradient_square_moment_hat)
                                                       + self.epsilon)
        
        if self.complex_params:
            params = real_imag_to_complex_vec(params)

        return params


### MODULE TESTS ###

def _test():
    """
    Run tests on the module.
    """
    # These are hand checked tests.
    adam = Adam()
    params = np.array([[0, 1],
                       [2, 3]])
    adam.initialize(params.shape, params.dtype)
    grads = np.array([[0, 1],
                      [2, 3]])
    params1 = np.array([[0,         0.99988889],
                        [1.99988889, 2.99988889]])
    params2 = np.array([[0,          0.99965432],
                        [1.99965432, 2.99965432]])
    assert(adam.update(grads, params).all() == params1.all())
    assert(adam.update(grads, params1).all()
           == params2.all())

    params = np.array([[1+2j, 3+4j],
                       [5+6j, 7+8j]])
    grads = np.array([[1+1j, 0+0j],
                      [0+0j, -1-1j]])
    adam.initialize(params.shape, params.dtype)
    params1 = adam.update(grads, params)
    assert(params1[0][1] == params[0][1] and params[1][0] == params1[1][0])

if __name__ == "__main__":
    _test()
        
        
        
        
