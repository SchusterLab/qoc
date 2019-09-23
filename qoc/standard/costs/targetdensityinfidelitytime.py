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
        # NOTE: Autograd doesn't support vjps of anp.trace with axis arguments.
        # Nor does it support the vjp of anp.einsum(...ii->..., a).
        # Therefore, we must use a for loop to index the traces.
        # The following computations are equivalent to:
        # inner_products = (anp.trace(anp.matmul(self.target_densities_dagger, densities),
        #                             axis1=-1, axis2=-2) / self.hilbert_size)
        prods = anp.matmul(self.target_densities_dagger, densities)
        fidelity_sum = 0
        for i, prod in enumerate(prods):
            inner_prod = anp.trace(prod)
            fidelity = anp.abs(inner_prod)
            fidelity_sum = fidelity_sum + fidelity
        fidelity_normalized = fidelity_sum / (self.density_count * self.hilbert_size)
        infidelity = 1 - fidelity_normalized
        cost_normalized = infidelity / self.cost_eval_count

        return cost_normalized * self.cost_multiplier
