"""
cost.py - This module defines the parent cost function class.
"""

from autograd.extend import Box
import autograd.numpy as anp

class Cost(object):
    """
    This class is the parent class for all cost functions.
    
    Fields:
    augmented_lagrangian :: bool - Whether or not to employ the augmented
        lagrangian method. Set to true if this cost has a constraint
        and the corresponding field is set in the optimization call.
    constraint :: float - the maximum tolerable cost, costs above this constraint
        will not be penalized
    cost_multiplier :: float - the weight factor for this cost
    cost_multiplier_scale :: float - if the augmented lagrangian method
        is employed, this is the factor by which the cost multiplier is
        scaled
    lagrange_multiplier :: float - this value is used by the
        augmented lagrangian method
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs to be computed
                                       at each optimization time step, False
                                       if it should be computed only at the
                                       final optimization time step
    """
    name = "parent_cost"
    requires_step_evaluation = False

    
    def __init__(self, constraint=None,
                 cost_multiplier=1.,
                 cost_multiplier_scale=1.2,):
        """
        See class definition for parameter specification.
        """
        super().__init__()
        self.augmented_lagrangian = False
        self.constraint = constraint
        self.cost_multiplier = cost_multiplier
        self.cost_multiplier_scale = cost_multiplier_scale
        self.lagrange_multiplier = 0


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
        
        if self.constraint is not None:
            cost_ = anp.maximum(cost_ - self.constraint, 0)
        if self.augmented_lagrangian:
            cost_ = (self.cost_multiplier * (cost_ ** 2)
                              + self.lagrange_multiplier * cost_)
        elif not self.cost_multiplier == 1.:
            cost_ = cost_ * cost_multiplier

        return cost_

        
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
    
    
    def update(self):
        pass
