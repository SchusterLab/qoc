"""
controlnorm.py - a module to define a cost on the
norm of controls
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ControlNorm(Cost):
    """
    a cost to penalize high control norms

    Fields:
    control_count :: int - the number of controls at each
        time step
    control_step_count :: int - the number of time steps
    cost_multiplier :: float - the weight factor for this cost
    max_control_norms :: ndarray (control_count) - the maximum norm for each control
    name :: str - a unique identifier for this cost
    nomalization_constant :: int - control_count * control_step_count
    """
    name = "control_norm"
    requires_step_evaluation = False

    def __init__(self, control_count, control_step_count,
                 max_control_norms, cost_multiplier=1.,):
        """
        See class fields for argument information.
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_count = control_count
        self.control_step_count = control_step_count
        self.max_control_norms = max_control_norms
        self.normalization_constant = control_count * control_step_count

    
    def cost(self, controls, states, system_step):
        """
        Args:
        controls :: ndarray (control_step_count, control_count)
            - the control parameters for all time steps
        states :: ndarray - an array of the initial states (or densities) evolved to
            the current time step
        system_step :: int - the system time step

        Returns:
        cost :: float - the penalty
        """
        # Normalize the controls.
        normalized_controls = anp.divide(controls, self.max_control_norms)

        # Penalize the sum of the square of the absolute value of the normalized
        # controls.
        cost = anp.sum(anp.square(anp.abs(normalized_controls)))
        cost_normalized = anp.divide(cost, self.normalization_constant)
        
        return self.cost_multiplier * cost_normalized


def _test():
    """
    Run test on the module.
    """
    controls = np.array(((1+2j, 3-4j), (5-6j, 7+8j), (9+10j, 11+12j),))
    control_count = controls.shape[1]
    control_step_count = controls.shape[0]
    max_control_norms = np.array((np.sqrt(181), np.sqrt(265),))
    cn = ControlNorm(control_count, control_step_count,
                      max_control_norms,)
    cost = cn.cost(controls, None, None)
    expected_cost = 0.48089926682650547
    
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _test()
