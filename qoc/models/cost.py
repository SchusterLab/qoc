"""
cost.py - This module defines the parent cost function class.
"""

class Cost(object):
    """
    This class is the parent class for all cost functions.
    
    Fields:
    cost_multiplier :: float - the weight factor for this cost
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
        raise NotImplementedError("The cost {} has not implemented an evaluation function."
                                  "".format(self))
