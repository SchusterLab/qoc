"""
cost.py - a module to define a class to encapsulate a cost funciton
"""

class Cost(object):
    """a class to encapsulate a cost function
    Fields:
    alpha :: float - the weight factor for this cost
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs to be computed
                                       at each optimization time step, False
                                       if it should be computed only at the
                                       final optimization time step
    """
    name = "parent_cost"
    requires_step_evaluation = False

    
    def __init__(self, alpha):
        """
        See class definition for parameter specification.
        """
        super().__init__()
        self.alpha = alpha


    def __str__(self):
        return self.name

    
    def __repr__(self):
        return self.__str__()

        
    def compute(step, states, all_params):
        """
        an autograd compatible function (https://github.com/HIPS/autograd)
        to compute the cost at each pulse time step given the pulse time step,
        the learning control parameters for all time steps, and the states at
        that time step
        Args:
        step :: int - the pulse time step
        states :: [numpy.ndarray] - a list of the initial states evolved to the
                                    current time step
        all_params :: numpy.ndarray - the control parameters for all time steps
        Returns:
        cost :: float - the cost for the given time step, parameters, and states
        """
        return 0.

    
