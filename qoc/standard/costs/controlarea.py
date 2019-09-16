"""
controlarea.py - This module defines a cost function that penalizes
the "area under the curve" of the control pulse.
"""

import autograd.numpy as anp

from qoc.models import (Cost,)

class ControlArea(Cost):
    """
    This cost penalizes the area under the
    function of time generated by the discrete control parameters.
    
    Fields:
    control_count
    cost_multiplier
    max_control_norms
    normalization_constant :: float - used to normalize the cost
    """

    def __init__(self, control_count,
                 max_control_norms,
                 cost_multiplier=1.):
        """
        See class fields for arguments not listed here.
        
        Args:
        control_step_count
        evolution_time
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_count = control_count
        self.max_control_norms = max_control_norms
        self.normalization_constant = control_count * control_step_count


    def cost(self, controls, states, system_step):
        """
        Compute the penalty.

        Args:
        controls
        states
        system_step
        
        Returns:
        cost
        """
        normalized_controls = controls / self.max_control_norms
        cost = 0
        for i in range(self.control_count):
            cost = cost + anp.abs(anp.sum(normalized_controls[:, i]))
        cost_normalized = cost / self.normalization_constant

        return cost_normalized * self.cost_multiplier
