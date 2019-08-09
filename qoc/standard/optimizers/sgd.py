"""
sgd.py - a module for defining the Stochastic Gradient Descent optimizer
"""

import numpy as np

from qoc.models.optimizer import Optimizer


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


    def run(self, args, function, iteration_count,
            initial_params, jacobian):
        """
        Run a SGD optimization series.
        Args:
        args :: any - a tuple of arguments to pass to the function
            and jacobian
        function :: any -> float
            - the function to minimize
        iteration_count :: int - how many iterations to perform
        initial_params :: numpy.ndarray - the initial optimization values
        jacobian :: numpy.ndarray - the jacobian of the function
            with respect to the params
        Returns: none
        """
        params = initial_params
        for i in range(iteration_count):
            grads = jacobian(params, *args)
            params = self.update(grads, params)


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
