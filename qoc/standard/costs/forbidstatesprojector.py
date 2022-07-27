"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""


import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose
from qoc.standard.functions import conjugate_transpose_m
import autograd.numpy as anp
from qoc.standard.functions import s_a_s_multi,block_fre
from scipy.sparse import  bmat
class ForbidStatesprojector(Cost):
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


    def __init__(self, projector,
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
        self.state_count = projector .shape[0]
        self.cost_evaluation_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.cost_normalization_constant = self.cost_evaluation_count * self.state_count
        self.projectors = projector
        self.forbidden_states_count=len(projector)
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
            for i, projector in enumerate(self.projectors):
                state = states[0]
                state_cost = 0
                state_dagger = conjugate_transpose_m(state)
                a = projector.dot(state)
                fidelity = np.real(np.matmul(state_dagger, a)[0][0])
                state_cost = state_cost + fidelity
                # ENDFOR
                state_cost_normalized = state_cost / self.forbidden_states_count
                cost = cost + state_cost_normalized
            # ENDFOR

            # Normalize the cost for the number of evolving states
            # and the number of times the cost is computed.
            cost_normalized = cost / self.cost_normalization_constant
        else:
            cost = 0
            for i, projector in enumerate(self.projectors):
                state = states
                state_cost = 0
                state_dagger=conjugate_transpose(state)
                a=anp.matmul(projector,state[0])
                fidelity = anp.real(anp.matmul(a, state_dagger )[0])
                state_cost = state_cost + fidelity
                # ENDFOR
                state_cost_normalized = state_cost / self.forbidden_states_count
                cost = cost + state_cost_normalized
            # ENDFOR

            # Normalize the cost for the number of evolving states
            # and the number of times the cost is computed.
            cost_normalized = cost / self.cost_normalization_constant

        return cost_normalized * self.cost_multiplier
    def average(self,psi,projector):
        return np.matmul(conjugate_transpose_m(psi),projector.dot(psi))
    def gradient_initialize(self, reporter):
        self.final_states=reporter.final_states[0]
        self.back_states = []
        for i in range(len(self.projectors)):
            self.back_states.append ((self.projectors[i]).dot(self.final_states))
        self.back_states=np.array(self.back_states)

    def update_state_back(self, A):
        for i in range(len(self.projectors)):
            self.back_states[i]=self.new_state[i]+(self.projectors[i]).dot(self.final_states)

    def update_state_forw(self, A,tol):
        self.final_states = s_a_s_multi(A,tol, self.final_states)

    def gradient(self, A,E,tol):
        grads = 0
        self.new_state=[]
        for i in range(len(self.projectors)):
            b_state, new_state = block_fre(A, E, tol, self.back_states[i])
            self.new_state.append(new_state)
            grads = grads + self.cost_multiplier * (2  * np.real(
                    np.matmul(conjugate_transpose_m(b_state), self.final_states))) /( self.state_count*self.cost_evaluation_count*self.forbidden_states_count)
        return grads