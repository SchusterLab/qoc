"""
controlarea.py - This module defines a cost function that penalizes
the "area under the curve" of the control parameters.
"""


import numpy as np

from qoc.models import (Cost,)

class ControlArea_manual(Cost):
    """
    This cost penalizes the area under the
    function of time generated by the discrete control parameters.
    
    Fields:
    control_count
    control_size
    cost_multiplier
    max_control_norms
    name
    requires_step_evaluation
    """
    name = "control_area"
    requires_step_evaluation = False

    def __init__(self, control_count,
                 control_eval_count,
                 cost_multiplier=1.,
                 max_control_norms=None,):
        """
        See class fields for arguments not listed here.
        
        Aruments:
        control_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_count = control_count
        self.control_size = control_count * control_eval_count
        self.max_control_norms = max_control_norms
        self.type="control"


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
            normalized_control = controls

        # The cost is the discrete integral of each normalized control parameter
        # over the evolution time.
        cost = 0
        self.sign = []
        for i in range(self.control_count):
            self.sign.append(np.sum(normalized_controls[:, i])/np.abs(np.sum(normalized_controls[:, i])))
            cost = cost + np.abs(np.sum(normalized_controls[:, i]))
        cost_normalized = cost / self.control_size

        return cost_normalized * self.cost_multiplier
    def gradient_initialize(self, reporter):
        return
    def update_state(self, propagator):
        return

    def gradient(self,controls,i,k):
        if self.max_control_norms is not None:
            return  self.sign[k]*self.cost_multiplier/((self.control_size)*self.max_control_norms)
        else:
            return self.sign[k]*self.cost_multiplier/(self.control_size)


