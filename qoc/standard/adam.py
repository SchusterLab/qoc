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
    apply_clip_grads :: bool
    apply_learning_rate_decay :: bool
    apply_scale_grads :: bool
    beta_1 :: float - gradient decay bias
    beta_2 :: float - gradient squared decay bias
    clip_grads :: float - the maximum absolute value at which the gradients
        should be element-wise clipped
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
    scale_grads :: float - the value to scale the norm of the gradients to
    """
    name = "adam"

    def __init__(self, beta_1=0.9, beta_2=0.999, clip_grads=None,
                 epsilon=1e-8, learning_rate=1e-3,
                 learning_rate_decay=None, scale_grads=None):
        """
        See class definition for argument specifications.
        Default values are chosen in accordance to those proposed
        in the paper.
        """
        super().__init__()
        if scale_grads is None:
            self.apply_scale_grads = False
        else:
            self.apply_scale_grads = True            
        if clip_grads is None:
            self.apply_clip_grads = False
        else:
            self.apply_clip_grads = True
        if learning_rate_decay is None:
            self.apply_learning_rate_decay = False
        else:
            self.apply_learning_rate_decay = True
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.clip_grads = clip_grads
        self.epsilon = epsilon
        self.gradient_moment = None
        self.gradient_square_moment = None
        self.initial_learning_rate = learning_rate
        self.iteration_count = 0
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.scale_grads = scale_grads


    def __str__(self):
        return ("{}, beta_1: {}, beta_2: {}, epsilon: {}, lr0: {}, "
                "lr_decay: {}, clip_grads: {}, scale_grads: {}"
                "".format(self.name, self.beta_1, self.beta_2,
                          self.epsilon, self.initial_learning_rate,
                          self.learning_rate_decay,
                          self.clip_grads, self.scale_grads))
    

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


    def update(self, grads, params):
        """Update the learning parameters for the current iteration.
        
        Implementation Note: I believe it is faster to check the modification
        conditionals each iteration than it would be to define seperate
        functions to do the same work and then call update_vanilla with
        modified parameters. Namely, because each function call requires a
        stack context change. It would be faster altogether if there was a
        seperate update function for each combination of modifications,
        but that would create duplicate code, which I want to avoid.
        If we really cared that much about performance we wouldn't be
        using python.
        Args:
        grads :: numpy.ndarray - the gradients of the cost function with
            respect to each learning parameter for the current iteration
        params :: numpy.ndarray - the learning parameters for the
            current iteration
        Returns:
        new_params :: numpy.ndarray - the learning parameters to be used
            for the next iteration
        """
        # Apply learning rate decay.
        if self.apply_learning_rate_decay:
            learning_rate = (self.initial_learning_rate
                             * np.exp(-np.divide(self.iteration_count,
                                                 self.learning_rate_decay)))
        else:
            learning_rate = self.initial_learning_rate

        # Apply gradient clipping.
        if self.apply_clip_grads:
            grads = np.clip(grads, -self.clip_grads, self.clip_grads)

        # Apply gradient scaling.
        if self.apply_scale_grads:
            grads_norm = np.linalg.norm(grads)
            grads = (grads / grads_norm) * self.scale_grads

        # Vanilla update procedure.
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


### MODULE TESTS ###

def _test():
    """
    Run tests on the module.
    """
    # These are hand checked tests.
    adam = Adam()
    params = np.array([[0, 1],
                       [2, 3]])
    grads = np.array([[0, 1],
                      [2, 3]])
    params1 = np.array([[0,         0.99988889],
                        [1.99988889, 2.99988889]])
    params2 = np.array([[0,          0.99965432],
                        [1.99965432, 2.99965432]])
    # assert(adam.update(grads, params).all() == params1.all())
    # assert(adam.update(grads, params1).all()
    #        == params2.all())

    # TOOD: rewrite test to map params.
    params = np.array([[1+2j, 3+4j],
                       [5+6j, 7+8j]])
    grads = np.array([[1+1j, 0+0j],
                      [0+0j, -1-1j]])
    params1 = adam.update(grads, params)
    # assert(params1[0][1] == params[0][1] and params[1][0] == params1[1][0])

if __name__ == "__main__":
    _test()
        
        
        
        
