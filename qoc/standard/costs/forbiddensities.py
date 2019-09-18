"""
forbiddensities.py - This module defines a cost function
that penalizes the occupation of a set of forbidden densities.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models.cost import Cost
from qoc.standard.functions.convenience import conjugate_transpose

class ForbidDensities(Cost):
    """
    This class penalizes the occupation of a set of forbidden densities.

    Fields:
    cost_multiplier
    cost_normalization_constant
    forbidden_densities_count
    forbidden_densities_dagger
    hilbert_size
    name
    requires_step_evaluation
    """
    name = "forbid_densities"
    requires_step_evaluation = True


    def __init__(self,
                 forbidden_densities,
                 system_eval_count,
                 cost_eval_step=1,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguemnts:
        cost_eval_step
        forbidden_densities
        system_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        density_count = forbidden_densities.shape[0]
        cost_evaluation_count, _ = np.divmod(system_eval_count - 1, cost_eval_step)
        self.cost_normalization_constant = cost_evaluation_count * density_count
        self.forbidden_densities_count = np.array([forbidden_densities_.shape[0]
                                                   for forbidden_densities_
                                                   in forbidden_densities])
        self.forbidden_densities_dagger = conjugate_transpose(forbidden_densities)
        self.hilbert_size = forbidden_densities.shape[3]


    def cost(self, controls, densities, system_eval_step):
        """
        Compute the penalty.

        Arguments:
        controls
        densities
        system_eval_step
        
        Returns:
        cost
        """
        # The cost is the overlap (fidelity) of the evolved density and each
        # forbidden density.
        cost = 0
        for i, forbidden_densities_dagger_ in enumerate(self.forbidden_densities_dagger):
            density = densities[i]
            density_cost = 0
            for forbidden_density_dagger in forbidden_densities_dagger_:
                inner_product = (anp.trace(anp.matmul(forbidden_density_dagger,
                                                      density)) / self.hilbert_size)
                fidelity = anp.real(inner_product * anp.conjugate(inner_product))
                density_cost = density_cost + fidelity
            #ENDFOR
            density_cost_normalized = density_cost / self.forbidden_densities_count[i]
            cost = cost + density_cost_normalized
        #ENDFOR
        
        # Normalize the cost for the number of evolving densities
        # and the number of times the cost is computed.
        cost_normalized = cost / self.cost_normalization_constant
        
        return cost_normalized * self.cost_multiplier


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
