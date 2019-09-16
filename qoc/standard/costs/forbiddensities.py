"""
forbiddensities.py - This module defines a cost function
to penalize the occupation
of forbidden density matrices.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models.cost import Cost
from qoc.standard.functions import conjugate_transpose

class ForbidDensities(Cost):
    """
    This class encapsulates a cost function that penalizes
    the occupation of forbidden densities.

    Fields:
    cost_multiplier
    density_normalization_constants :: ndarray - the number of densities
        that each evolving density is forbidden from
    hilbert_size :: int - the dimension of the hilbert space
    forbidden_densities_dagger :: ndarray - the conjugate transpose of
        the forbidden densities
    name :: str - a unique identifier for this cost
    normalization_constant :: int - used to normalize the cost
    requires_step_evaluation :: bool - True if the cost needs
        to be computed at each optimization time step, False
        if it should be computed only at the final optimization
        time step
    """
    name = "forbid_densities"
    requires_step_evaluation = True


    def __init__(self, forbidden_densities, system_step_count, cost_multiplier=1.):
        """
        See class fields for arguments not listed here.

        Args:
        forbidden_densities :: ndarray - an array where each entry
            in the first axis is an array of densities that the corresponding
            evolving density is forbidden from, that is, each evolving
            density has its own list of forbidden densities
        system_step_count :: int - the number of evolution steps
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.forbidden_densities_dagger = conjugate_transpose(forbidden_densities)
        self.density_normalization_constants = np.array([density_forbidden_densities.shape[0]
                                                         for density_forbidden_densities
                                                         in forbidden_densities])
        self.hilbert_size = forbidden_densities.shape[-1]
        density_count = forbidden_densities.shape[0]
        self.normalization_constant = density_count * system_step_count


    def cost(self, controls, densities, system_step):
        """
        Args:
        controls :: ndarray - the control parameters for all time steps
        densities :: ndarray - an array of the initial densities evolved to
            the current time step
        system_step :: int - the system time step
        Returns:
        cost :: float - the penalty
        """
        cost = 0
        # Compute the fidelity for each evolution density and its forbidden densities.
        for i, density_forbidden_densities_dagger in enumerate(self.forbidden_densities_dagger):
            density = densities[i]
            density_cost = 0
            for forbidden_density_dagger in density_forbidden_densities_dagger:
                inner_product = (anp.trace(anp.matmul(forbidden_density_dagger,
                                                      density)) / self.hilbert_size)
                density_cost = density_cost + anp.square(anp.abs(inner_product))
            #ENDFOR
            cost = cost + anp.divide(density_cost, self.density_normalization_constants[i])
        #ENDFOR
        
        # Normalize the cost for the number of evolving densities
        # and the number of time evolution steps.
        cost = (cost / self.normalization_constant)
        
        return self.cost_multiplier * cost


def _test():
    """
    Run tests on the module.
    """
    system_step_count = 10
    state0 = np.array([[1], [0]])
    density0 = np.matmul(state0, conjugate_transpose(state0))
    forbid0_0 = np.array([[1], [0]])
    density0_0 = np.matmul(forbid0_0, conjugate_transpose(forbid0_0))
    forbid0_1 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    density0_1 = np.matmul(forbid0_1, conjugate_transpose(forbid0_1))
    state1 = np.array([[0], [1]])
    density1 = np.matmul(state1, conjugate_transpose(state1))
    forbid1_0 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    density1_0 = np.matmul(forbid1_0, conjugate_transpose(forbid1_0))
    forbid1_1 = np.divide(np.array([[1j], [1j]]), np.sqrt(2))
    density1_1 = np.matmul(forbid1_1, conjugate_transpose(forbid1_1))
    densities = np.stack((density0, density1,))
    forbidden_densities0 = np.stack((density0_0, density0_1,))
    forbidden_densities1 = np.stack((density1_0, density1_1,))
    forbidden_densities = np.stack((forbidden_densities0, forbidden_densities1,))
    fd = ForbidDensities(forbidden_densities, system_step_count)
    
    cost = fd.cost(None, densities, None)
    expected_cost = 7 / 640
    assert(np.allclose(cost, expected_cost,))


if __name__ == "__main__":
    _test()
