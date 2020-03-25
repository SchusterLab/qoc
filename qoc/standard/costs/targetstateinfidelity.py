"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import autograd.numpy as anp
from autograd.extend import Box
import copy
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import (
    conjugate_transpose, rms_norm
)

class TargetStateInfidelity(Cost):
    """
    This cost penalizes the infidelity of an evolved state
    and a target state.

    Fields:
    constraint
    cost_multiplier
    cost_multiplier_step
    lagrange_multiplier
    name
    requires_step_evaluation
    rms :: bool - whether or not to use the rms norm
    state_count
    target_states_dagger
    """
    name = "target_state_infidelity"
    requires_step_evaluation = False

    def __init__(self, target_states, constraint=None,
                 cost_multiplier=1.,
                 cost_multiplier_step=None,
                 rms=False):
        """
        See class fields for arguments not listed here.
        
        Arguments:
        target_states
        """
        super().__init__(constraint=constraint,
                         cost_multiplier=cost_multiplier,
                         cost_multiplier_step=cost_multiplier_step)
        self.rms = rms
        self.state_count = target_states.shape[0]
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
        if self.rms:
            cost_ = rms_norm(states - self.target_states_dagger)
        else:
            # The cost is the infidelity of each evolved state and its target state.
            inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
            fidelities = anp.real(inner_products * anp.conjugate(inner_products))
            fidelity_normalized = anp.sum(fidelities) / self.state_count
            cost_ = 1 - fidelity_normalized

        augmented_cost = self.augment_cost(cost_)
        if not isinstance(cost_._value, Box):
            print("tsic: {}, tsia: {}"
                  "".format(cost_._value,
                            augmented_cost._value))

        return augmented_cost
