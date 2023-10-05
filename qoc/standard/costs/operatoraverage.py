"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the average value of an operator.
"""

import numpy as np
from qoc.models import Cost
import autograd as ad

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
    requires_step_evaluation = True
    type = "control_implicit_related"

    def __init__(self, operator,  cost_multiplier=1., ):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.operator = operator
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
        # The cost is the infidelity of each evolved state and its target state.
        if states.shape[1] > 1:
            raise Exception("This cost contribution is only for single state transfer")
        control_eval_count = len(controls[0])
        self.grads_factor = self.cost_multiplier / ( control_eval_count )
        if gradients_method == "AD" or gradients_method == "SAD":
            import autograd.numpy as np
        else:
            import numpy as np
        state_dagger = np.conjugate(np.transpose(states))

        self.operator_states = np.transpose(np.matmul(self.operator,states))
        cost = np.trace(np.real(np.matmul(state_dagger, np.matmul(self.operator,states))))
        self.cost_value = cost * self.grads_factor
        if gradients_method == "SAD":
            def cost_function(states):
                state_dagger = np.conjugate(np.transpose(states))
                cost = np.trace(np.real(np.matmul(state_dagger, np.matmul(self.operator,states))))
                cost_value = cost * self.grads_factor
                return cost_value
            self.cost_value, SAD_bps = ad.value_and_grad(cost_function)(states)
            self.SAD_bps.append(1 / 2 * SAD_bps.conjugate().transpose())
        return cost * self.grads_factor

    def gradient_initialize(self, ):
        """

        Returns
        -------

        """
        if len(self.SAD_bps) == 0:
            return self.operator_states* self.grads_factor
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
            return self.operator.dot(states).transpose()* self.grads_factor
        else:
            return_state = self.SAD_bps[-1]
            del self.SAD_bps[-1]
            return return_state

