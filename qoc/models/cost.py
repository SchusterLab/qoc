"""
cost.py - This module defines the parent cost function class.
"""

from autograd.extend import Box
import autograd.numpy as anp

class Cost(object):
    """
    This class is the parent class for all cost functions.
    
    Fields:
    constraint :: float - the maximum tolerable cost, costs above this constraint
        will not be penalized
    cost_multiplier :: float - the weight factor for this cost
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs to be computed
                                       at each optimization time step, False
                                       if it should be computed only at the
                                       final optimization time step
    should_augment :: bool
    """
    name = "parent_cost"
    requires_step_evaluation = False
    
    def __init__(self, constraint=None,
                 cost_multiplier=1.,):
        """
        See class definition for parameter specification.
        """
        super().__init__()
        self.constraint = constraint
        self.cost_multiplier = cost_multiplier
        self.should_augment = constraint is not None


    def __str__(self):
        return self.name

    
    def __repr__(self):
        return self.__str__()


    def augment_cost(self, cost_):
        """
        Employ the augmented lagrangian method for a constrained problem, or
        the lagrange multiplier method for an unconstrained problem.
        
        Arguments:
        cost_ :: float - the base, normalized cost
        
        Returns:
        augmented_cost :: float - the augmented cost
        
        References:
        [0] https://en.wikipedia.org/wiki/Augmented_Lagrangian_method
        """
        if self.constraint is not None and self.should_augment:
            # Only add a cost if it is greater than the constraint.
            cost_ = anp.maximum(cost_ - self.constraint, 0)
            augmented_cost = cost_ * self.cost_multiplier
        else:
            # augmented_cost = cost_ * self.cost_multiplier
            augmented_cost = cost_

        return augmented_cost

        
    def cost(self, params, states, step):
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

