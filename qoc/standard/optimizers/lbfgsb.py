"""
lbfgsb.py - a module to expose the L-BFGS-B optimization algorithm
"""

from scipy.optimize import minimize
from qoc.models.optimizer import Optimizer

class LBFGSB(Optimizer):
    """
    The L-BFGS-B optimizer.
    Fields: none
    """

    def __init__(self):
        """
        See class docstring for argument information.
        """
        super().__init__()


    def run(self, args, function, iteration_count, 
            initial_params, jacobian):
        """
        Run the L-BFGS-B method.
        Args:
        args :: any - a tuple of arguments to pass to the function
            and jacobian
        function :: any -> float
            - the function to minimize
        iteration_count :: int - how many iterations to perform
        initial_params :: numpy.ndarray - the initial optimization values
        jacobian :: numpy.ndarray - the jacobian of the function with respect
            to the params
        Returns:
        result :: scipy.optimize.OptimizeResult
        """
        options = {
            "maxiter": iteration_count,
        }
        return minimize(function, initial_params, args=args,
                        method="L-BFGS-B", jac=jacobian,
                        options=options)
        
        
