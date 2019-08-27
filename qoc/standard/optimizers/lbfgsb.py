"""
lbfgsb.py - a module to expose the L-BFGS-B optimization algorithm
"""

from scipy.optimize import minimize

class LBFGSB(object):
    """
    The L-BFGS-B optimizer.
    Fields: none
    """

    def __init__(self):
        """
        See class docstring for argument information.
        """
        super().__init__()


    def run(self, function, iteration_count, 
            initial_params, jacobian, args=()):
        """
        Run the L-BFGS-B method.

        Args:
        args :: any - a tuple of arguments to pass to the function
            and jacobian
        function :: any -> float
            - the function to minimize
        iteration_count :: int - how many iterations to perform
        initial_params :: numpy.ndarray - the initial optimization values
        jacobian :: any -> numpy.ndarray - a function that returns the jacobian
            of `function`

        Returns:
        result :: scipy.optimize.OptimizeResult
        """
        options = {
            "maxiter": iteration_count,
        }
        return minimize(function, initial_params, args=args,
                        method="L-BFGS-B", jac=jacobian,
                        options=options)
