"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the average value of an operator.
"""

import numpy as np
from qoc.models import Cost


class OperatorAverage(Cost):
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
    name = "operatoraverage"
    requires_step_evaluation = False
    type = "control_implicit_related"

    def __init__(self, operator,  cost_multiplier=1., ):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.operator = operator

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
        if len(states) > 1:
            raise Exception("This cost contribution is only for single state transfer")
        control_eval_count = len(controls[0])
        self.grads_factor = self.cost_multiplier / ( control_eval_count )
        if gradients_method == "AD":
            import autograd.numpy as np
        else:
            import numpy as np
        state_dagger = np.conjugate(np.transpose(states))
        self.operator_states = self.operator.dot(states)
        cost = np.real(np.matmul(state_dagger, self.operator.dot(states)))

        return cost * self.grads_factor

    def gradient_initialize(self, ):
        """

        Returns
        -------

        """
        return self.operator_states

    def update_state_back(self, states):
        """

        Parameters
        ----------
        states :

        Returns
        -------

        """
        return self.operator.dot(states)

