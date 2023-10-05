"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import numpy as np
import autograd as ad
from qoc.models import Cost

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
    target_states
    grads_factor
    inner_products_sum
    type
    """
    name = "target_state_infidelity"
    requires_step_evaluation = False
    type = "control_implicit_related"

    def __init__(self, target_states,neglect_relative_phase=False, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.
        
        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        self.target_states = target_states
        self.target_states_dagger = np.conjugate(target_states)
        self.neglect_relative_phase = neglect_relative_phase
        self.grads_factor = -self.cost_multiplier / self.state_count ** 2
        self.SAD_bps = None

    def cost(self, controls, states, gradients_method):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        gradients_method

        Returns:
        cost
        """
        # The cost is the infidelity of each evolved state and its target state.
        if gradients_method == "AD" or gradients_method == "SAD":
            import autograd.numpy as np
        else:
            import numpy as np
        inner_products = np.matmul(self.target_states_dagger, states)
        if self.neglect_relative_phase==False:
            self.inner_products_sum = np.trace(inner_products)
            fidelity = np.real(
                self.inner_products_sum * np.conjugate(self.inner_products_sum)) / self.state_count ** 2
        else:
            fidelity = np.trace(np.abs(inner_products)**2)
            fidelity = fidelity / self.state_count ** 2
        infidelity = 1 - fidelity
        self.cost_value = infidelity * self.cost_multiplier
        if gradients_method == "SAD":
            def cost_function(states):
                inner_products = np.matmul(self.target_states_dagger, states)
                self.inner_products_sum = np.trace(inner_products)
                fidelity = np.real(
                    self.inner_products_sum * np.conjugate(self.inner_products_sum)) / self.state_count ** 2
                return (1 - fidelity)* self.cost_multiplier
            self.cost_value, self.SAD_bps = ad.value_and_grad(cost_function)(states)
            self.SAD_bps = 1/2 * np.transpose(self.SAD_bps.conjugate())
        return self.cost_value

    def gradient_initialize(self,):
        """

        Returns
        -------

        """
        if type(self.SAD_bps) != type(None):
            return self.SAD_bps
        else:
            return self.target_states * self.inner_products_sum * self.grads_factor

    def update_state_back(self, states):
        """

        Parameters
        ----------
        states :

        Returns
        -------

        """
        return np.zeros_like(self.target_states_dagger)

