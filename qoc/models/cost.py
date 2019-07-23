"""
cost.py - a module to define a class to encapsulate a cost funciton
"""

class Cost(object):
    """a class to encapsulate a cost function
    Fields:
    requires_step_evaluation :: bool - True if the cost needs to be computed
                                       at each optimization time step, False
                                       if it should be computed only at the
                                       final optimization time step
    Methods:
    compute :: (step :: int, all_params :: numpy.ndarray,
                states :: [numpy.ndarray]) -> cost :: float
      - an autograd compatible function (https://github.com/HIPS/autograd)
        to compute the cost at each pulse time step given the pulse time step,
        the learning control parameters for all time steps, and the states at
        that time step
    """
    requires_step_evaluation = False
    
    def __init__(self):
        super()

        
    def compute(step, all_params, states):
        """
        See class definition for function specification.
        Args:
        step :: int - the pulse time step
        all_params :: numpy.ndarray - the control parameters for all time steps
        states :: [numpy.ndarray] - a list of the initial states evolved to the
                                    current time step
        Returns:
        cost :: float - the cost for the given time step, parameters, and states
        """
        return 0.

    
