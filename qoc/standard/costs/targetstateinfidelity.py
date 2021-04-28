"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose,matmuls
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

    def __init__(self, target_states, cost_multiplier=1.):
        """
        See class fields for arguments not listed here.
        
        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        self.target_states_dagger = conjugate_transpose(target_states)
        self.target_states  =target_states
        self.type = "non-control"

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
        # The cost is the infidelity of each evolved state and its target state.
        self.inner_products = anp.matmul(self.target_states_dagger,states)[:, 0, 0]
        fidelities = anp.real(self.inner_products * anp.conjugate(self.inner_products))
        fidelity_normalized = anp.sum(fidelities) / self.state_count
        infidelity = 1 - fidelity_normalized
        
        return infidelity * self.cost_multiplier

    def gradient_initialize(self,reporter):
        self.final_states = reporter.final_states
        self.back_states=np.zeros_like(self.target_states,dtype="complex_")
        for i in range(self.state_count):
            self.back_states[i]=self.target_states[i]* self.inner_products[i]

    def update_state(self,propagator):
        self.final_states=matmuls(propagator,self.final_states)
        self.back_states=matmuls(propagator,self.back_states)

    def gradient(self, dt, Hk):
        grads=0
        for i in range(self.state_count):
            grads=grads+self.cost_multiplier*(-2 * dt * np.imag(anp.matmul(conjugate_transpose(self.back_states[i]), anp.matmul(Hk, self.final_states[i])) ))/ self.state_count
        return grads



