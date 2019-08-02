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
    beta_1 :: float - gradient decay bias
    beta_2 :: float - gradient squared decay bias
    epsilon :: float - fuzz factor
    gradient_moment :: numpy.ndarray - running optimization variable
    gradient_square_moment :: numpy.ndarray - running optimization variable
    initial_learning_rate :: float - the initial step size
    iteration_count :: int - the current count of iterations performed
    learning_rate :: float - the current step size
    learning_rate_decay :: float - the number of iterations it takes for
        the learning rate to decay by 1/e, if not specified, no decay is
        applied
    name :: str - identifier for the optimizer
    update :: (self, grads :: numpy.ndarray, params :: numpy.ndarray)
               -> new_params :: numpy.ndarray
        - the parameter update method
    """
    name = "adam"

    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 learning_rate=1e-3, learning_rate_decay=None):
        """
        See class definition for argument specifications.
        Default values are chosen in accordance to those proposed
        in the paper.
        """
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.gradient_moment = None
        self.gradient_square_moment = None
        self.initial_learning_rate = learning_rate
        self.iteration_count = 0
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        if learning_rate_decay is None:
            self.update = self.update_vanilla
        else:
            self.update = self.update_decay


    def __str__(self):
        return ("{}, beta_1: {}, beta_2: {}, epsilon: {}, lr: {}"
                "".format(self.name, self.beta_1, self.beta_2,
                          self.epsilon, self.learning_rate,))
    

    def run(self, args, function, iteration_count,
            initial_params, jacobian):
        """
        Run an Adam optimization series.
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
        self.iteration_count = 0
        self.gradient_moment = np.zeros_like(initial_params)
        self.gradient_square_moment = np.zeros_like(initial_params)
        
        params = initial_params
        for i in range(iteration_count):
            grads = jacobian(params, *args)
            params = self.update(grads, params)


    def update_vanilla(self, grads, params):
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
        self.iteration_count += 1
        self.gradient_moment = (self.beta_1 * self.gradient_moment
                                + (1 - self.beta_1) * grads)
        self.gradient_square_moment = (self.beta_2 * self.gradient_square_moment
                                       + (1 - self.beta_2) * np.square(grads))
        gradient_moment_hat = np.divide(self.gradient_moment,
                                        1 - np.power(self.beta_1, self.iteration_count))
        gradient_square_moment_hat = np.divide(self.gradient_square_moment,
                                               1 - np.power(self.beta_2, self.iteration_count))
        return params + self.learning_rate * np.divide(gradient_moment_hat,
                                                       np.sqrt(gradient_square_moment_hat)
                                                       + self.epsilon)

    
    def update_decay(self, grads, params):
        """Update the learning parameters for the current iteration.
        Use learning rate decay.
        Args:
        grads :: numpy.ndarray - the gradients of the cost function with
            respect to each learning parameter for the current iteration
        params :: numpy.ndarray - the learning parameters for the
            current iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters to be used
            for the next iteration
        """
        self.learning_rate = (self.initial_learning_rate
                              * np.exp(self.iteration_count
                                       / self.learning_rate_decay))
        
        return self.update_vanilla(grads, params)
    
    


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

    # TOOD: rewrite test to map params.
    params = np.array([[1+2j, 3+4j],
                       [5+6j, 7+8j]])
    grads = np.array([[1+1j, 0+0j],
                      [0+0j, -1-1j]])
    adam.initialize(params.shape, params.dtype)
    params1 = adam.update(grads, params)
    assert(params1[0][1] == params[0][1] and params[1][0] == params1[1][0])

if __name__ == "__main__":
    _test()
        
        
        
        
