"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""


import numpy as np
import autograd as ad
from qoc.models import Cost

class ForbidStates(Cost):
    """
    This cost penalizes the occupation of a set of forbidden states.

    Fields:
    cost_multiplier
    cost_normalization_constant
    forbidden_states
    inner_products
    grads_factor
    forbidden_states_dagger
    name
    type
    requires_step_evalution
    """
    name = "forbid_states"
    requires_step_evaluation = True
    type = "control_implicit_related"

    def __init__(self, forbidden_states,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        cost_eval_step
        forbidden_states
        system_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.forbidden_states_dagger = np.conjugate(forbidden_states.transpose())
        self.forbidden_states = forbidden_states
        self.state_count = np.count_nonzero(forbidden_states)
        self.SAD_bps = []

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
        # The cost is the overlap (fidelity) of the evolved state and each
        # forbidden state.
        if gradients_method == "AD" or gradients_method == "SAD":
            import autograd.numpy as np
        else:
            import numpy as np
        control_eval_count = len(controls[0])
        state_count = len(states)
        self.grads_factor = self.cost_multiplier/ (state_count * control_eval_count)
        self.inner_products = np.sum(np.abs(np.multiply(self.forbidden_states_dagger, states))**2)
        fidelity = self.inner_products
        if gradients_method == "SAD":
            def cost_function(states):
                self.inner_products = np.matmul(self.forbidden_states_dagger, states)
                fidelity = np.trace(np.abs(self.inner_products) ** 2)
                return self.grads_factor * fidelity
            self.cost_value, self.SAD_bps = ad.value_and_grad(cost_function)(states)
            self.SAD_bps.append(1 / 2 * np.transpose(self.SAD_bps.conjugate()))
        return self.grads_factor * fidelity

    def gradient_initialize(self,):
        """

        Returns
        -------

        """
        if len(self.SAD_bps) == 0:
            states_return = np.zeros_like(self.forbidden_states, dtype=np.complex128)
            for i in range(len(self.inner_products)):
                states_return[i] = self.inner_products[i][i] * self.forbidden_states[i] * self.grads_factor
            return states_return
        else:
            return_state = self.SAD_bps[-1]
            del self.SAD_bps[-1]
            return return_state

    def update_state_back(self, states):
        """

        Parameters
        ----------
        states :

        Returns
        -------

        """
        if len(self.SAD_bps) == 0:
            states_return = np.zeros_like(self.forbidden_states, dtype=np.complex128)
            inner_products = np.matmul(self.forbidden_states_dagger, states)
            for i in range(len(inner_products)):
                states_return[i] = inner_products[i][i] * self.forbidden_states[i] * self.grads_factor
            return states_return
        else:
            return_state = self.SAD_bps[-1]
            del self.SAD_bps[-1]
            return return_state
