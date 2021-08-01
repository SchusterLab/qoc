"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""


import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose,matmuls
import autograd.numpy as anp
class ForbidStates(Cost):
    """
    This cost penalizes the occupation of a set of forbidden states.

    Fields:
    cost_multiplier
    cost_normalization_constant
    forbidden_states_count
    forbidden_states_dagger
    name
    requires_step_evalution
    """
    name = "forbid_states"
    requires_step_evaluation = True


    def __init__(self, forbidden_states,
                 system_eval_count,
                 cost_eval_step=1,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        cost_eval_step
        forbidden_states
        system_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = forbidden_states.shape[0]
        self.cost_evaluation_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.cost_normalization_constant = self.cost_evaluation_count * self.state_count
        self.forbidden_states_count = np.array([forbidden_states_.shape[0]
                                                for forbidden_states_
                                                in forbidden_states])
        self.forbidden_states_dagger = conjugate_transpose(forbidden_states)
        self.forbidden_states=forbidden_states
        self.type = "non-control"

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
        # The cost is the overlap (fidelity) of the evolved state and each
        # forbidden state.
        if manual_mode==True:
            cost = 0
            self.inner_products=[]
            for i, forbidden_states_dagger_ in enumerate(self.forbidden_states_dagger):
                state = states[i]
                state_cost = 0
                self.inner_products.append([])
                for forbidden_state_dagger in forbidden_states_dagger_:
                    inner_product = np.matmul(forbidden_state_dagger, state)[0, 0]
                    fidelity = np.real(inner_product * np.conjugate(inner_product))
                    state_cost = state_cost + fidelity
                    self.inner_products[i].append(inner_product)
            # ENDFOR
                state_cost_normalized = state_cost / self.forbidden_states_count[i]
                cost = cost + state_cost_normalized

            #ENDFOR
        
            # Normalize the cost for the number of evolving states
            # and the number of times the cost is computed.
            cost_normalized = cost / self.cost_normalization_constant
        else:
            cost = 0
            for i, forbidden_states_dagger_ in enumerate(self.forbidden_states_dagger):
                state = states[i]
                state_cost = 0
                for forbidden_state_dagger in forbidden_states_dagger_:
                    inner_product = anp.matmul(forbidden_state_dagger, state)[0, 0]
                    fidelity = anp.real(inner_product * anp.conjugate(inner_product))
                    state_cost = state_cost + fidelity
                # ENDFOR
                state_cost_normalized = state_cost / self.forbidden_states_count[i]
                cost = cost + state_cost_normalized
            # ENDFOR

            # Normalize the cost for the number of evolving states
            # and the number of times the cost is computed.
            cost_normalized = cost / self.cost_normalization_constant
        
        return cost_normalized * self.cost_multiplier


    def gradient_initialize(self, reporter):
        self.final_states=reporter.final_states
        self.back_states = np.zeros_like(self.forbidden_states, dtype="complex_")
        for i in range(len(self.inner_products)):
            for j in range(len(self.inner_products[i])):
                self.back_states[i] = self.forbidden_states[i][j] * self.inner_products[i][j]

    def update_state_back(self, propagator):
        self.inner_products = np.zeros_like(self.inner_products)
        for i in range(len(self.inner_products)):
            for j in range(len(self.inner_products[i])):
                self.inner_products[i][j]=np.matmul(self.forbidden_states_dagger[j], self.final_states[i])
                self.back_states[i][j]=np.matmul(propagator, self.back_states[i][j])+self.inner_products[i][j]*self.forbidden_states[i][j]

    def update_state_forw(self, propagator):
        self.final_states = matmuls(propagator, self.final_states)

    def gradient(self, dt, Hk):
        grads = 0
        for i in range(len(self.inner_products)):
            for j in range(len(self.inner_products[i])):
                grads = grads + self.cost_multiplier * (2 * dt * np.real(
                    np.matmul(conjugate_transpose(self.back_states[i][j]), np.matmul(Hk, self.final_states[i])))) /( self.state_count*self.cost_evaluation_count*self.forbidden_states_count[i])

        return grads