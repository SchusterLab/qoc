"""
targetdensityinfidelity.py - This module defines the target density
infidelity cost function.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetDensityInfidelity(Cost):
    """a class to encapsulate the target density infidelity cost function
    Fields:
    cost_multiplier :: float - the wieght factor for this cost
    density_count :: int - the number of evolving densities
    hilbert_size :: int - the dimension of the hilbert space
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs
        to be computed at each optimization time step, False
        if it should be computed only at the final optimization
        time step

    target_densities_dagger :: ndarray (density_count x hilbert_size x hilbert_size)
        - the hermitian conjugate of the target densities
    """
    name = "target_density_infidelity"
    requires_step_evaluation = False

    def __init__(self, target_densities, cost_multiplier=1.):
        """
        See class definition for parameter specification.
        target_densities :: ndarray (density_count x hilbert_size x hilbert_size)
            - an array of densities
            that correspond to the target density for each of the initial densities
            used in optimization
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.density_count = target_densities.shape[0]
        self.hilbert_size = target_densities.shape[-1]
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
        return self.cost_multiplier * infidelity


def _tests():
    """
    Run test on the module.
    """
    density0 = np.array([[0], [1]])
    target0 = np.array([[1], [0]])
    densities = np.stack((density0,), axis=0)
    targets = np.stack((target0,), axis=0)
    ti = TargetDensityInfidelity(targets)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 1))

    ti = TargetDensityInfidelity(densities)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 0))

    density0 = np.array([[1], [0]])
    density1 = (np.array([[1j], [1]]) / np.sqrt(2))
    target0 = np.array([[1j], [0]])
    target1 = np.array([[1], [0]])
    densities = np.stack((density0, density1,), axis=0)
    targets = np.stack((target0, target1,), axis=0)
    ti = TargetDensityInfidelity(targets)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, .25))


if __name__ == "__main__":
    _tests()
