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
    control_weights :: ndarray (control_eval_count x control_count)
        - These weights, each of which should be no greater than 1,
        represent the factor by which each control's magnitude is penalized.
        If no weights are specified, each control's magnitude is penalized
        equally.
    controls_size
    cost_multiplier
    max_control_norms
    name
    requires_step_evaluation
    """
    name = "control_norm"
    requires_step_evaluation = False

    def __init__(self, control_count,
                 control_eval_count,
                 control_weights=None,
                 cost_multiplier=1.,
                 max_control_norms=None,):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_count
        control_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_weights = control_weights
        self.controls_size = control_eval_count * control_count
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
        # Normalize the controls.
        if self.max_control_norms is not None:
            controls = controls / self.max_control_norms
            
        # Weight the controls.
        if self.control_weights is not None:
            controls = controls[:,] * self.control_weights

        # The cost is the sum of the square of the modulus of the normalized,
        # weighted, controls.
        cost = anp.sum(anp.real(controls * anp.conjugate(controls)))
        cost_normalized = cost / self.controls_size
        
        return cost_normalized * self.cost_multiplier
