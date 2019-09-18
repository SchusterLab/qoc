"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

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
        inner_products = anp.matmul(self.target_states_dagger, states)[:, 0, 0]
        fidelities = anp.real(inner_products * anp.conjugate(inner_products))
        fidelity_normalized = anp.sum(fidelities) / self.state_count
        infidelity = 1 - fidelity_normalized
        
        return infidelity * self.cost_multiplier


def _tests():
    """
    Run test on the module.
    """
    state0 = np.array([[0], [1]])
    target0 = np.array([[1], [0]])
    states = np.stack((state0,), axis=0)
    targets = np.stack((target0,), axis=0)
    ti = TargetStateInfidelity(targets)
    cost = ti.cost(None, states, None)
    assert(np.allclose(cost, 1))

    ti = TargetStateInfidelity(states)
    cost = ti.cost(None, states, None)
    assert(np.allclose(cost, 0))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    target0 = np.array([[1j], [0]])
    target1 = np.array([[1], [0]])
    states = np.stack((state0, state1,), axis=0)
    targets = np.stack((target0, target1,), axis=0)
    ti = TargetStateInfidelity(targets)
    cost = ti.cost(None, states, None)
    assert(np.allclose(cost, .25))


if __name__ == "__main__":
    _tests()
