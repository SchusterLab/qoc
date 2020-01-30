"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions.convenience import conjugate_transpose

class ForbidStates(Cost):
    """
    This cost penalizes the occupation of a set of forbidden states.

    Fields:
    cost_multiplier
    cost_normalization_constant
    forbidden_states_count
    forbidden_states_dagger
    name
    requires_step_evalution
    """
    name = "forbid_states"
    requires_step_evaluation = True


    def __init__(self, forbidden_states,
                 system_eval_count,
                 cost_eval_step=1,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        cost_eval_step
        forbidden_states
        system_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        state_count = forbidden_states.shape[0]
        cost_evaluation_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.cost_normalization_constant = cost_evaluation_count * state_count
        self.forbidden_states_count = np.array([forbidden_states_.shape[0]
                                                for forbidden_states_
                                                in forbidden_states])
        self.forbidden_states_dagger = conjugate_transpose(forbidden_states)


    def cost(self, controls, states, system_eval_step):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        system_eval_step

        Returns:
        cost
        """
        # The cost is the overlap (fidelity) of the evolved state and each
        # forbidden state.
        cost = 0
        for i, forbidden_states_dagger_ in enumerate(self.forbidden_states_dagger):
            state = states[i]
            state_cost = 0
            for forbidden_state_dagger in forbidden_states_dagger_:
                inner_product = anp.matmul(forbidden_state_dagger, state)[0, 0]
                fidelity = anp.real(inner_product * anp.conjugate(inner_product))
                state_cost = state_cost + fidelity
            #ENDFOR
            state_cost_normalized = state_cost / self.forbidden_states_count[i]
            cost = cost + state_cost_normalized
        #ENDFOR
        
        # Normalize the cost for the number of evolving states
        # and the number of times the cost is computed.
        cost_normalized = cost / self.cost_normalization_constant
        
        return cost_normalized * self.cost_multiplier
