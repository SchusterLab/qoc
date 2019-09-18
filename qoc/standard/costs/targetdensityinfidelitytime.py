"""
targetdensityinfidelitytime.py - This module defines a cost function
that penalizes the infidelity of evolved densities and their
respective target densities at each cost evaluation step.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetDensityInfidelityTime(Cost):
    """
    This class penalizes the infidelity of evolved states
    and their respective target states at each cost evaluation step.
    The intended result is that a lower infidelity is
    achieved earlier in the system evolution.

    Fields:
    cost_eval_count
    cost_multiplier
    density_count
    hilbert_size
    name
    requires_step_evaluation
    target_densities_dagger
    """
    name = "target_density_infidelity_time"
    requires_step_evaluation = False

    def __init__(self, system_eval_count, target_densities,
                 cost_eval_step=1, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_densities
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.cost_eval_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.density_count = target_densities.shape[0]
        self.hilbert_size = target_densities.shape[1]
        self.target_densities_dagger = conjugate_transpose(np.stack(target_densities))


    def cost(self, controls, densities, sytem_eval_step):
        """
        Compute the penalty.

        Arguments:
        controls
        densities
        system_eval_step

        Returns:
        cost
        """
        # The cost is the infidelity of each evolved density and its target density.
        inner_products = (anp.trace(anp.matmul(self.target_densities_dagger, densities)[:, 0, 0],
                                    axis1=-1, axis2=-2) / self.hilbert_size)
        fidelities = anp.real(inner_products * anp.conjugate(inner_products))
        fidelity_normalized = anp.sum(fidelities) / self.density_count
        infidelity = 1 - fidelity_normalized
        # Normalize the cost for the number of times the cost is evaluated.
        cost_normalized = infidelity / self.cost_eval_count

        return cost_normalized * self.cost_multiplier


def _tests():
    """
    Run test on the module.
    """
    system_step_count = 10
    state0 = np.array([[0], [1]])
    density0 = np.matmul(state0, conjugate_transpose(state0))
    target_state0 = np.array([[1], [0]])
    target_density0 = np.matmul(target_state0, conjugate_transpose(target_state0))
    densities = np.stack((density0,), axis=0)
    targets = np.stack((target_density0,), axis=0)
    ti = TargetDensityInfidelityTime(system_step_count, targets)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 0.1))

    ti = TargetDensityInfidelity(system_step_count, densities)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 0.075))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    density0 = np.matmul(state0, conjugate_transpose(state0))
    density1 = np.matmul(state1, conjugate_transpose(state1))
    target_state0 = np.array([[1j], [0]])
    target_state1 = np.array([[1], [0]])
    target_density0 = np.matmul(target_state0, conjugate_transpose(target_state0))
    target_density1 = np.matmul(target_state1, conjugate_transpose(target_state1))
    densities = np.stack((density0, density1,), axis=0)
    targets = np.stack((target_density0, target_density1,), axis=0)
    ti = TargetDensityInfidelity(system_step_count, targets)
    cost = ti.cost(None, densities, None)
    expected_cost = (1 - np.divide(5, 32)) / 10
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _tests()
