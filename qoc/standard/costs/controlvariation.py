"""
controlvariation.py - This module defines a cost function
that penalizes variations of the control parameters.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ControlVariation(Cost):
    """
    This cost penalizes the variations of the control parameters
    from one `control_eval_step` to the next.

    Fields:
    cost_multiplier
    name
    order
    requires_step_evaluation
    type
    """
    name = "control_variation"
    requires_step_evaluation = False
    type = "control_explicit_related"
    def __init__(self,
                 cost_multiplier=1.,
                 order=1):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_count
        control_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.order = order


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

        # Penalize the square of the absolute value of the difference
        # in value of the control parameters from one step to the next.
        diffs = anp.diff(controls, axis=0, n=self.order)
        cost = anp.sum(anp.real(diffs * anp.conjugate(diffs)))

        return cost * self.cost_multiplier
