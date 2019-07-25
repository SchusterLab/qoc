"""
adam.py - a module for defining the Adam optimizer
"""

import numpy as np

from qoc.models import Optimizer

class Adam(Optimizer):
    """
    a class to define the Adam optimizer
    This implementation follows the original algorithm
    https://arxiv.org/abs/1412.6980.
    Fields:
    learning_rate :: float - the initial step size
    beta_1 :: float - gradient decay bias
    beta_2 :: float - gradient squared decay bias
    epsilon :: float - fuzz factor
    """
    name = "adam"

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                   epsilon=1e-8):
        """
        See class definition for argument specifications.
        Default values are chosen in accordance to those proposed
        in the paper.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon


    def __str__(self):
        return ("{}, lr: {}, beta_1: {}, beta_2: {}, epsilon: {}"
                "".format(self.name, self.learning_rate, self.beta_1,
                          self.beta_2, self.epsilon))

    
    def initialize(self, params_shape):
        """Initialize the optimizer for a new optimization series.
        Arguments:
        params_shape :: tuple(int) - the shape of the learning parameters
        Returns: none
        """
        self.iteration_count = 0
        self.gradient_moment = np.zeros(params_shape)
        self.gradient_square_moment = np.zeros(params_shape)


    def update(self, params, grads):
        """Update the learning parameters for the current iteration.
        Args:
        params :: numpy.ndarray - the learning parameters for the
            current iteration
        grads :: numpy.ndarray - the gradients of the cost function with
            respect to each learning parameter for the current iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters to be used
            for the next iteration
        """
        self.iteration_count += 1
        self.gradient_moment = (self.beta_1 * self.gradient_moment
                                + (1 - self.beta_1) * grads)
        self.gradient_square_moment = (self.beta_2 * self.gradient_square_moment
                                       + (1 - self.beta_2) * np.square(grads))
        gradient_moment_hat = np.divide(self.gradient_moment,
                                        1 - np.power(self.beta_1, self.iteration_count))
        gradient_square_moment_hat = np.divide(self.gradient_square_moment,
                                               1 - np.power(self.beta_2, self.iteration_count))
        return params - learning_rate * np.divide(gradient_moment_hat,
                                                  np.sqrt(gradient_square_moment_hat)
                                                  * self.epsilon)
        
        
        
        
