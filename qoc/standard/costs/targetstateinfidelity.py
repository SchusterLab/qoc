"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import autograd.numpy as anp
import numpy as np
from functools import partial
from qoc.models import Cost
from scqubits.utils.cpu_switch import get_map_method
import multiprocessing
from qoc.standard.functions import conjugate_transpose
from qoc.standard.functions import conjugate_transpose_m
from qoc.standard.functions import s_a_s_multi,block_fre,krylov
import scqubits.settings as settings
from qoc.standard.functions import column_vector_list_to_matrix
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

        self.target_states = target_states
        self.target_states_dagger = conjugate_transpose(self.target_states)
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
            inner_products=anp.matmul(self.target_states_dagger,states)
            inner_products_sum=anp.sum(anp.trace(inner_products))
            fidelity=anp.real(inner_products_sum * anp.conjugate(inner_products_sum)) / self.state_count ** 2
            infidelity = 1 - fidelity
            #inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
            #inner_products_sum = anp.sum(inner_products)
            #fidelity_normalized = anp.real(
                #inner_products_sum * anp.conjugate(inner_products_sum)) / self.state_count ** 2
            #infidelity = 1 - fidelity_normalized
        return infidelity* self.cost_multiplier

    def gradient_initialize(self, reporter):
        if self.neglect_relative_phase == False:
            self.final_states = reporter.final_states
            self.back_states = self.target_states * self.inner_products_sum
        else:
            self.final_states = reporter.final_states
            self.back_states = np.zeros_like(self.target_states, dtype="complex_")
            for i in range(self.state_count):
                self.back_states[i] = self.target_states[i] * self.inner_products[i]

    def update_state_forw(self, A,tol):
        if len(self.final_states) >= 2:
            n = multiprocessing.cpu_count()
            func = partial(s_a_s_multi, A, tol)
            settings.MULTIPROC = "pathos"
            map = get_map_method(n)
            states_mul = []
            for i in range(len(self.final_states)):
                states_mul.append(self.final_states[i])
            self.final_states = np.array(map(func, states_mul))
        else:
            self.final_states = krylov(A, tol, self.final_states)
    def update_state_back(self, A):
        self.back_states = self.new_state

    def gradient(self, A,E,tol):
        if len(self.final_states) >= 100:
            n = multiprocessing.cpu_count()
            func = partial(block_fre, A, E, tol)
            settings.MULTIPROC = "pathos"
            map = get_map_method(n)
            states_mul = []
            for i in range(len(self.back_states)):
                states_mul.append(self.back_states[i])
            states = map(func, states_mul)
            b_state = np.zeros_like(self.back_states)
            self.new_state = np.zeros_like(self.back_states)
            for i in range(len(states)):
                b_state[i] = states[i][0]
                self.new_state[i] = states[i][1]
            grads = 0
            if self.neglect_relative_phase == False:
                for i in range(self.state_count):
                    a=np.matmul(conjugate_transpose_m(b_state[i]), self.final_states[i])
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        a)) / (
                                    self.state_count ** 2)
            else:
                for i in range(self.state_count):
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose_m(b_state[i]), self.final_states[i]))) / (
                                self.state_count)
        else:
            grads = 0
            self.new_state = []
            if self.neglect_relative_phase == False:
                for i in range(self.state_count):
                    b_state, new_state = block_fre(A, E, tol, self.back_states[i])
                    self.new_state.append(new_state)
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose_m(b_state), self.final_states[i]))) / (
                                    self.state_count ** 2)
            else:
                for i in range(self.state_count):
                    b_state, new_state = block_fre(A, E, tol, self.back_states[i])
                    self.new_state.append(new_state)
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose_m(b_state), self.final_states[i]))) / (
                                self.state_count)
        return grads



