"""
targetstateinfidelitytime.py - This module defins a cost function that
penalizes the infidelity of evolved states and their respective target states
at each cost evaluation step.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

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


    def __init__(self, system_eval_count, target_states,
                 cost_eval_step=1, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.
        
        Arguments:
        target_states
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.cost_eval_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.state_count = target_states.shape[0]
        self.target_states_dagger = conjugate_transpose(anp.stack(target_states))


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
        # Normalize the cost for the number of times the cost is evaluated.
        cost_normalized = infidelity / self.cost_eval_count

        return cost_normalized * self.cost_multiplier


def _tests():
    """
    Run test on the module.
    """
    step_count = 10
    state0 = np.array([[0], [1]])
    target0 = np.array([[1], [0]])
    states = np.stack((state0,), axis=0)
    targets = np.stack((target0,), axis=0)
    ti = TargetStateInfidelityTime(step_count, targets)
    cost = ti.cost(None, states, None)
    expected_cost = 0.1
    assert(np.allclose(cost, expected_cost))

    ti = TargetStateInfidelityTime(step_count, states)
    cost = ti.cost(None, states, None)
    expected_cost = 0
    assert(np.allclose(cost, expected_cost))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    target0 = np.array([[1j], [0]])
    target1 = np.array([[1], [0]])
    states = np.stack((state0, state1,), axis=0)
    targets = np.stack((target0, target1,), axis=0)
    ti = TargetStateInfidelityTime(step_count, targets)
    cost = ti.cost(None, states, None)
    expected_cost = 0.025
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _tests()
