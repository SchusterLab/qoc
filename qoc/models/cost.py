"""
cost.py - a module to define a class to encapsulate a cost funciton
"""

from autograd import elementwise_grad, jacobian
import autograd.numpy as anp
import numpy as np

class Cost(object):
    """a class to encapsulate a cost function
    Fields:
    cost_multiplier :: float - the weight factor for this cost
    dcost_dparams :: (params :: numpy.ndarray, states :: numpy.ndarray, step :: int)
                      -> dcost_dparams :: numpy.ndarray
        - the gradient of the cost function with respect to the parameters
    dcost_dstates :: (params :: numpy.ndarray, states :: numpy.ndarray, step :: int)
                      -> dcost_dstates :: numpy.ndarray
        - the gradient of the cost function with respect to the states
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs to be computed
                                       at each optimization time step, False
                                       if it should be computed only at the
                                       final optimization time step
    """
    name = "parent_cost"
    requires_step_evaluation = False
    
    def __init__(self, cost_multiplier=1.):
        """
        See class definition for parameter specification.
        """
        super().__init__()
        self.cost_multiplier = cost_multiplier

        # Define the gradient of the cost function.
        cost_wrapper = (lambda *args, **kwargs:
                        anp.real(self.cost(*args, **kwargs)))
        self.dcost_dparams = (lambda *args, **kwargs:
                              jacobian(cost_wrapper, 0)
                              (*args, **kwargs))
        self.dcost_dstates = (lambda *args, **kwargs:
                              jacobian(cost_wrapper, 1)
                              (*args, **kwargs))


    def __str__(self):
        return self.name

    
    def __repr__(self):
        return self.__str__()

        
    def cost(params, states, step):
        """
        an autograd compatible function (https://github.com/HIPS/autograd)
        to compute the cost at each pulse time step given the pulse time step,
        the learning control parameters for all time steps, and the states at
        that time step
        Args:
        params :: numpy.ndarray - the control parameters for all time steps
        states :: numpy.ndarray - an array of the initial states evolved to the
            current time step
        step :: int - the pulse time step
        Returns:
        cost :: float - the cost for the given parameters, states, and time step
        """
        return 0.
