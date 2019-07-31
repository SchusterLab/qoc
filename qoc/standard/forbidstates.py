"""
forbidstates.py - a module to encapsulate the forbidden states cost function
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.util import conjugate_transpose

class ForbidStates(Cost):
    """a class to encapsulate the forbid states cost function
    Fields:
    alpha :: float - the wieght factor for this cost
    dcost_dparams :: (params :: numpy.ndarray, states :: numpy.ndarray, step :: int)
                      -> dcost_dparams :: numpy.ndarray
        - the gradient of the cost function with respect to the parameters
    dcost_dstates :: (params :: numpy.ndarray, states :: numpy.ndarray, step :: int)
                      -> dcost_dstates :: numpy.ndarray
        - the gradient of the cost function with respect to the states
    forbidden_states_dagger :: numpy.ndarray - the conjugate transpose of
        the forbidden states

    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs
        to be computed at each optimization time step, False
        if it should be computed only at the final optimization
        time step

    state_normalization_constants :: np.ndarray - the number of states
        that each evolving state is forbidden from
    step_normalization_constant :: float - the number of evolution steps
    total_state_normalization_constant :: float - the total number
        of evolving states
    """
    name = "forbid_states"
    requires_step_evaluation = True


    def __init__(self, forbidden_states, step_count, alpha=1.):
        """
        See class definition for parameter specification.
        Args:
        forbidden_states :: numpy.ndarray - an array where each entry
            in the first axis is an array of states that the corresponding
            evolving state is forbidden from, that is, each evolving
            state has its own list of forbidden states
        step_count :: int - the total number of steps in an evolution
        """
        super().__init__(alpha)
        # This cost function does not make use of parameter penalties.
        self.dcost_dparams = (lambda params, states, step:
                              np.zeros_like(params))
        self.forbidden_states_dagger = conjugate_transpose(forbidden_states)
        self.state_normalization_constants = np.array([len(state_forbidden_states)
                                                       for state_forbidden_states
                                                       in forbidden_states])
        self.step_normalization_constant = step_count
        self.total_state_normalization_constant = len(forbidden_states)


    def cost(self, params, states, step):
        """
        Args:
        params :: numpy.ndarray - the control parameters for all time steps
        states :: numpy.ndarray - an array of the initial states evolved to
            the current time step
        step :: int - the pulse time step
        Returns:
        cost :: float - the penalty
        """
        cost = 0
        # Compute the fidelity for each evolution state and its forbidden states.
        for i, state_forbidden_states_dagger in enumerate(self.forbidden_states_dagger):
            state = states[i]
            state_cost = 0
            for forbidden_state_dagger in state_forbidden_states_dagger:
                state_cost = cost + anp.abs(anp.matmul(forbidden_state_dagger, state)[0,0])
            #ENDFOR
            cost = cost + state_cost * self.state_normalization_constants[i]
        #ENDFOR
        
        # Normalize the cost for the number of evolving states
        # and the number of time evolution steps.
        cost = (cost / (self.total_state_normalization_constant
                        * self.step_normalization_constant))
        
        return self.alpha * cost
