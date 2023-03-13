"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetStateInfidelity(Cost):
    """
    This cost penalizes the infidelity of an evolved state
    and a target state.

    Fields:
    cost_multiplier
    name
    requires_step_evaluation
    state_count
    target_states_dagger
    neglect_relative_phase
    """
    name = "target_state_infidelity"
    requires_step_evaluation = False

    def __init__(self, target_states,neglect_relative_phase=False, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.
        
        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        self.target_states_dagger = np.conjugate(target_states)
        self.neglect_relative_phase = neglect_relative_phase

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
        # The cost is the infidelity of each evolved state and its target state.
        inner_products = anp.matmul(self.target_states_dagger, states)
        if self.neglect_relative_phase==False:
            inner_products_sum = anp.sum(anp.trace(inner_products))
        else:
            inner_products_sum = anp.sum(anp.trace(anp.abs(inner_products)))
        fidelity = anp.real(inner_products_sum * anp.conjugate(inner_products_sum)) / self.state_count ** 2
        infidelity = 1 - fidelity
        return infidelity * self.cost_multiplier
