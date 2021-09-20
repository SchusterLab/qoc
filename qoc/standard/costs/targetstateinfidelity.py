"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose
from qoc.standard.functions import conjugate_transpose_m
from qoc.standard.functions import krylov,block_fre


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

    def __init__(self, target_states, cost_multiplier=1., neglect_relative_phase=False):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        self.target_states_dagger = conjugate_transpose(target_states)
        self.target_states = target_states
        self.type = "non-control"
        self.neglect_relative_phase = neglect_relative_phase
    def cost(self, controls, states, system_eval_step,manual_mode):
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
        if manual_mode==True:
            if self.neglect_relative_phase == False:
                inner_products = np.matmul(self.target_states_dagger, states)[:, 0, 0]
                self.inner_products_sum = np.sum(inner_products)
                fidelity_normalized = np.real(
                    self.inner_products_sum * np.conjugate(self.inner_products_sum)) / self.state_count ** 2
                infidelity = 1 - fidelity_normalized
            else:
                self.inner_products = np.matmul(self.target_states_dagger, states)[:, 0, 0]
                fidelities = np.real(self.inner_products * np.conjugate(self.inner_products))
                fidelity_normalized = np.sum(fidelities) / self.state_count
                infidelity = 1 - fidelity_normalized
        else:
            inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
            inner_products_sum = anp.sum(inner_products)
            fidelity_normalized = anp.real(
                inner_products_sum * anp.conjugate(inner_products_sum)) / self.state_count ** 2
            infidelity = 1 - fidelity_normalized
        return infidelity * self.cost_multiplier

    def gradient_initialize(self, reporter):
        if self.neglect_relative_phase == False:
            self.final_states = reporter.final_states
            self.back_states = self.target_states * self.inner_products_sum
        else:
            self.final_states = reporter.final_states
            self.back_states = np.zeros_like(self.target_states, dtype="complex_")
            for i in range(self.state_count):
                self.back_states[i] = self.target_states[i] * self.inner_products[i]

    def update_state_forw(self, A):
        self.final_states = krylov(A, self.final_states)

    def update_state_back(self, A):
        self.back_states = self.new_state

    def gradient(self, A,E,tol):
        grads = 0
        self.new_state = []
        if self.neglect_relative_phase == False:
            for i in range(self.state_count):
                b_state,new_state=block_fre(A, E, self.back_states[i], tol)
                self.new_state.append(new_state)
                a=conjugate_transpose_m(b_state)
                b= self.final_states[i]
                grads = grads + self.cost_multiplier * (-2 * np.real(
                    np.matmul(conjugate_transpose_m(b_state), self.final_states[i]))) / (
                                    self.state_count ** 2)
        else:
            for i in range(self.state_count):
                b_state, new_state = block_fre(A, E, self.back_states[i], tol)
                self.new_state.append(new_state)
                grads = grads + self.cost_multiplier * (-2 * np.real(
                    np.matmul(conjugate_transpose_m(b_state), self.final_states[i]))) / (
                                    self.state_count )
        return grads



