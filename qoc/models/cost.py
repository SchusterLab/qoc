"""
cost.py - This module defines the parent cost function class.
"""

import autograd.numpy as anp

class Cost(object):
    """
    This class is the parent class for all cost functions.
    
    Fields:
    constraint :: float - the maximum tolerable penalty, if set, the cost
        will employ the augmented lagrangian method for constrained optimization
    cost_multiplier :: float - the weight factor for this cost
    cost_multiplier_step :: float - the increment to the cost multiplier
        applied each iteration
    lagrange_multiplier :: float - the lagrange multiplier used in the
        augmented lagrangian method
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
                 cost_multiplier=1.,
                 cost_multiplier_step=None):
        """
        See class definition for parameter specification.
        """
        super().__init__()
        self.constraint = constraint
        self.cost_multiplier = cost_multiplier
        self.cost_multiplier_step = cost_multiplier_step
        self.lagrange_multiplier = 0
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
            # Compute the augmented cost.
            augmented_cost = ((self.cost_multiplier / 2) * (cost_ ** 2)
                              + self.lagrange_multiplier * cost_)
            # Update lagrange multiplier.
            self.lagrange_multiplier = self.lagrange_multiplier + self.cost_multiplier * cost_
        else:
            augmented_cost = cost_ * self.cost_multiplier

        # Update cost multiplier.
        if self.cost_multiplier_step is not None:
            self.cost_multiplier = self.cost_multiplier + self.cost_multiplier_step

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

