"""
targetstateinfidelitytime.py - This module defins a cost function that
penalizes the infidelity of evolved states and their respective target states
at each cost evaluation step.
"""


import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose
import autograd.numpy as anp
from qoc.standard.functions import krylov,block_fre
from scipy.sparse import bmat
class TargetStateInfidelityTime(Cost):
    """
    This cost penalizes the infidelity of evolved states
    and their respective target states at each cost evaluation step.
    The intended result is that a lower infidelity is
    achieved earlier in the system evolution.

    Fields:
    cost_eval_count
    cost_multiplier
    name
    requires_step_evaluation
    state_count
    target_states_dagger
    """
    name = "target_state_infidelity_time"
    requires_step_evaluation = True


    def __init__(self, system_eval_count, target_states,neglect_relative_phase=False,
                 cost_eval_step=1, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.cost_eval_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.state_count = target_states.shape[0]
        self.target_states_dagger = conjugate_transpose(np.stack(target_states))
        self.target_states = target_states
        self.type="non-control"
        self.inner_products_sum=[]
        self.neglect_relative_phase=neglect_relative_phase

    def cost(self, controls, states, system_eval_step,manual_mode=None):
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
                if len(self.inner_products_sum)==self.cost_eval_count :
                    self.inner_products_sum=[]
                inner_products = np.matmul(self.target_states_dagger, states)[:, 0, 0]
                inner_products_sum = np.sum(inner_products)
                self.inner_products_sum.append(inner_products_sum)
                fidelity_normalized = np.real(inner_products_sum * np.conjugate(inner_products_sum)) / self.state_count ** 2
                infidelity = 1 - fidelity_normalized
                # Normalize the cost for the number of times the cost is evaluated.
                cost_normalized = infidelity / self.cost_eval_count
            else:
                self.inner_products = np.matmul(self.target_states_dagger, states)[:, 0, 0]
                fidelities = np.real(self.inner_products * np.conjugate(self.inner_products))
                fidelity_normalized = np.sum(fidelities) / self.state_count
                infidelity = 1 - fidelity_normalized
                # Normalize the cost for the number of times the cost is evaluated.
                cost_normalized = infidelity / self.cost_eval_count
        else:
            inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
            inner_products_sum = anp.sum(inner_products)
            fidelity_normalized = anp.real(
                inner_products_sum * anp.conjugate(inner_products_sum)) / self.state_count ** 2
            infidelity = 1 - fidelity_normalized
            # Normalize the cost for the number of times the cost is evaluated.
            cost_normalized = infidelity / self.cost_eval_count
        return cost_normalized * self.cost_multiplier
    def gradient_initialize(self, reporter):
        if self.neglect_relative_phase == False:
            self.final_states = reporter.final_states
            self.back_states = self.target_states * self.inner_products_sum[self.cost_eval_count-1]
            self.i=self.cost_eval_count-2
        else:
            self.final_states = reporter.final_states
            self.back_states = np.zeros_like(self.target_states, dtype="complex_")
            for i in range(self.state_count):
                self.back_states[i] = self.target_states[i] * self.inner_products[i]


    def update_state_back(self, dt, A):
        if self.neglect_relative_phase == False:
            self.back_states = krylov(dt, A, self.back_states)
            for i in range(self.state_count):
                self.back_states[i] = self.back_states[i]+self.inner_products_sum[self.i]*self.target_states[i]
            self.i=self.i-1
        else:
            self.inner_products = np.matmul(self.target_states_dagger, self.final_states)[:, 0, 0]
            self.back_states = krylov(dt, A, self.back_states)
            for i in range(self.state_count):
                self.back_states[i] = self.back_states[i] + self.inner_products[i] * self.target_states[i]
    def update_state_forw(self, dt, A):
        if self.neglect_relative_phase == False:
            self.final_states = krylov(dt, A, self.final_states)
        else:
            self.final_states = krylov(dt, A, self.final_states)

    def gradient(self, dt, A, E,tol):
        grads = 0
        if self.neglect_relative_phase == False:
            for i in range(self.state_count):
                grads = grads + self.cost_multiplier * (-2  * np.real(
                    np.matmul(conjugate_transpose(self.back_states[i]), block_fre(dt,A,E,self.final_states[i],tol)))) /(( self.state_count**2)*self.cost_eval_count)
        else:
            for i in range(self.state_count):
                grads = grads + self.cost_multiplier * (-2  * np.real(
                    np.matmul(conjugate_transpose(self.back_states[i]), block_fre(dt,A,E,self.final_states[i],tol)))) / (
                                    self.state_count * self.cost_eval_count)
        return grads
