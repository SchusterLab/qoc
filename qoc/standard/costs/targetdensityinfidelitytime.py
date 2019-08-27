"""
targetdensityinfidelitytime.py - This module defines the target density
infidelity cost function computed each step.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetDensityInfidelityTime(Cost):
    """
    This class encapsulates the target density infidelity cost function
    computed each system step.
    The penalty of this function is calculated using the Frobenius inner product.

    Fields:
    cost_multiplier :: float - the wieght factor for this cost
    density_count :: int - the number of evolving densities
    hilbert_size :: int - the dimension of the hilbert space
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs
        to be computed at each optimization time step, False
        if it should be computed only at the final optimization
        time step
    system_step_count :: int - the 
    target_densities_dagger :: ndarray (density_count x hilbert_size x hilbert_size)
        - the hermitian conjugate of the target densities
    """
    name = "target_density_infidelity_time"
    requires_step_evaluation = False

    def __init__(self, system_step_count, target_densities, cost_multiplier=1.):
        """
        See class definition for arguments not listed here.

        Args:
        target_densities :: ndarray (density_count x hilbert_size x hilbert_size)
            - an array of densities
            that correspond to the target density for each of the initial densities
            used in optimization
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.density_count = target_densities.shape[0]
        self.hilbert_size = target_densities.shape[-1]
        self.system_step_count = system_step_count
        self.target_densities_dagger = conjugate_transpose(np.stack(target_densities))


    def cost(self, controls, densities, sytem_step):
        """
        Args:
        controls :: ndarray (control_step_count x control_count)
            - the control parameters for all time steps
        densities :: ndarray (density_count x hilbert_size x hilbert_size)
            - an array of the densities evolved to
            the current time step
        system_step :: int - the system time step

        Returns:
        cost :: float - the penalty
        """
        hilbert_size = self.hilbert_size
        fidelity = 0
        for i, target_density_dagger in enumerate(self.target_densities_dagger):
            density = densities[i]
            inner_product = anp.trace(anp.matmul(target_density_dagger, density))
            inner_product_normalized = inner_product / hilbert_size
            fidelity = fidelity + anp.square(anp.abs(inner_product_normalized))
        infidelity = 1 - (fidelity / self.density_count)
        infidelity_normalized = infidelity / self.system_step_count
        return self.cost_multiplier * infidelity


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
