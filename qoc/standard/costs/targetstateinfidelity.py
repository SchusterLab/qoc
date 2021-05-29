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
    """
    name = "target_state_infidelity"
    requires_step_evaluation = False

    def __init__(self, target_states,neglect_relative_pahse=False, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.
        
        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        self.target_states_dagger = conjugate_transpose(target_states)
        self.neglect_relative_pahse=neglect_relative_pahse

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
        if self.neglect_relative_pahse==False:
            inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
            inner_products_sum = anp.sum(inner_products)
            fidelity_normalized = anp.real(inner_products_sum * anp.conjugate(inner_products_sum)) / self.state_count ** 2
            infidelity = 1 - fidelity_normalized
        else:
            self.inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
            fidelities = anp.real(self.inner_products * anp.conjugate(self.inner_products))
            fidelity_normalized = anp.sum(fidelities) / self.state_count
            infidelity = 1 - fidelity_normalized
        
        return infidelity * self.cost_multiplier
