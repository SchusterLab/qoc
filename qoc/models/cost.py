"""
cost.py - a module to define a class to encapsulate a cost funciton
"""

class Cost(object):
    """a class to encapsulate a cost function
    Fields:
<<<<<<< HEAD
    alpha :: float - the weight factor for this cost
    name :: str - a unique identifier for this cost
=======
>>>>>>> 318831567b7dd9196600b9274c11d12a071d3af0
    requires_step_evaluation :: bool - True if the cost needs to be computed
                                       at each optimization time step, False
                                       if it should be computed only at the
                                       final optimization time step
<<<<<<< HEAD
    """
    name = "parent_cost"
    requires_step_evaluation = False

    
    def __init__(self, alpha):
        """
        See class definition for parameter specification.
        """
        super().__init__()
        self.alpha = alpha
=======
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
>>>>>>> 318831567b7dd9196600b9274c11d12a071d3af0

        
    def compute(step, all_params, states):
        """
<<<<<<< HEAD
        an autograd compatible function (https://github.com/HIPS/autograd)
        to compute the cost at each pulse time step given the pulse time step,
        the learning control parameters for all time steps, and the states at
        that time step
=======
        See class definition for function specification.
>>>>>>> 318831567b7dd9196600b9274c11d12a071d3af0
        Args:
        step :: int - the pulse time step
        all_params :: numpy.ndarray - the control parameters for all time steps
        states :: [numpy.ndarray] - a list of the initial states evolved to the
                                    current time step
        Returns:
        cost :: float - the cost for the given time step, parameters, and states
        """
        return 0.

    
