"""
controlnorm.py - This module defines a cost function that penalizes
the value of the norm of the control parameters.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ControlNorm(Cost):
    """
    This cost penalizes the value of the norm of the control parameters.

    Fields:
    control_weights :: ndarray (control_count x control_eval_count)
        - These weights, each of which should be no greater than 1,
        represent the factor by which each control's magnitude is penalized.
        If no weights are specified, each control's magnitude is penalized
        equally.
    cost_multiplier
    name
    requires_step_evaluation
    """
    name = "control_norm"
    requires_step_evaluation = False

    def __init__(self,
                 control_weights=None,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:

        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_weights = control_weights

    
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
        # Weight the controls.
        if self.control_weights is not None:
            controls = controls[:,] * self.control_weights

        # The cost is the sum of the square of the modulus of the normalized,
        # weighted, controls.
        cost = anp.sum(anp.real(controls * anp.conjugate(controls)))
        return cost * self.cost_multiplier
