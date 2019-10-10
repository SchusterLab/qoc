"""
controlnorm.py - This module defines a cost function that penalizes
the value of the norm of the control parameters.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class Speedup(Cost):
    """
    This cost penalizes the time it takes to complete the desired pulse.

    Fields:
    cost_multiplier
    name
    requires_step_evaluation
    state_count
    target_states_dagger
    """
    name = "speedup"
    requires_step_evaluation = True

    def __init__(self, target_states, system_eval_count,
                 cost_eval_step=1,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        system_eval_count
        cost_eval_step
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        cost_evaluation_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.cost_normalization_constant = cost_evaluation_count * self.state_count
        self.target_states_dagger = conjugate_transpose(target_states)

    
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
        # The cost is unity minus the overlap of the initial states with the target states.
        # However because of the way the costs are computed, we compute the overlap,
        # sum over all overlaps (over all steps), and then take unity minus the 
        # overlap later.
        inner_products = anp.matmul(target_state_dagger, states)[:,0,0]
        fidelities = anp.real(inner_products * anp.conjugate(inner_products))
        fidelity_normalized = anp.sum(fidelities) / self.cost_normalization_constant
        
        return fidelity_normalized * self.cost_multiplier
