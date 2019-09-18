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
    control_size
    cost_multiplier
    max_control_norms
    name
    requires_step_evaluation
    """
    name = "control_norm"
    requires_step_evaluation = False

    def __init__(self, control_count,
                 control_eval_count,
                 cost_multiplier=1.,
                 max_control_norms=None,):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_count
        control_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_size = control_count * control_step_count
        self.max_control_norms = max_control_norms

    
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
        if self.max_control_norms is not None:
            normalized_controls = controls / self.max_control_norms
        else:
            normalized_controls = controls

        # The cost is the sum of the square of the modulus of the normalized
        # controls.
        cost = anp.sum(anp.real(normalized_controls * anp.conjugate(normalized_controls)))
        cost_normalized = cost / self.control_size
        
        return cost_normalized * self.cost_multiplier
