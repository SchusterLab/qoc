"""
controlbandwidth.py - This module defines a cost function that penalizes
the difference between the maximum and minimum value taken by the control
over the evolution time.
"""

import autograd.numpy as anp

from qoc.models import (Cost, OperationPolicy)

class ControlBandwidth(Cost):
    """
    This cost penalizes the difference between
    the maximum and minimum value taken by the control
    over the evolution time.
    
    Fields:
    control_count
    cost_multiplier
    max_control_norms
    normalized_max_control_bandwidths
    """

    def __init__(self, control_count,
                 max_control_bandwidths,
                 max_control_norms,
                 cost_multiplier=1.):
        """
        See class docstring for arguments not listed here.
        
        Args:
        max_control_bandwidths :: ndarray (control_count)
            - an array that specifies the maximum bandwith that each control
            is allowed to take over the evolution time
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_count = control_count
        self.max_control_norms = max_control_norms
        self.normalized_max_control_bandwidths = max_control_bandwidths / (2 * max_control_norms)


    def cost(self, controls, states, system_step,
             operation_policy=OperationPolicy.CPU):
        """
        Compute the penalty.

        Args:
        controls
        operation_policy
        states
        system_step
        
        Returns:
        cost
        """
        control_count = self.control_count
        max_control_norms = self.max_control_norms
        normalized_controls = controls / max_control_norms
        
        if operation_policy == OperationPolicy.CPU:
            cost = 0
            for i in range(control_count):
                max_control_norm = max_control_norms[i]
                normalized_control = normalized_controls[:, i]
                normalized_max_control_bandwidth = self.normalized_max_control_bandwidths[i]
                normalized_max_control_value = anp.max(normalized_control)
                normalized_min_control_value = anp.min(normalized_control)
                normalized_control_bandwidth = ((normalized_max_control_value
                                                 - normalized_min_control_value) / 2)
                if normalized_control_bandwidth >= normalized_max_control_bandwidth:
                    control_cost = anp.square(anp.abs(normalized_control_bandwidth
                                                      - normalized_max_control_bandwidth))
                else:
                    control_cost = 0
                cost = cost + control_cost
        else:
            raise ValueError("This cost function does not support "
                             "the operation policy {}."
                             "".format(operation_policy))
        normalized_cost = cost / control_count
        
        return normalized_cost * self.cost_multiplier
