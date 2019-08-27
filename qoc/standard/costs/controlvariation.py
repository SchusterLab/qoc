"""
controlvariation.py - This module defines a cost function
that penalizes variations in control parameters.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ControlVariation(Cost):
    """
    This cost penalizes the rapid variations of control parameters.

    Fields:
    control_count :: int - the number of controls at each
        time step
    control_step_count :: int - the number of time steps,
        we require control_step_count >= order + 1
    cost_multiplier :: float - the weight factor for this cost
    max_control_norms :: ndarray (control_count) - the maximum norms
        for each control for all time
    name :: str - a unique identifier for this cost
    normalization_constant :: float - used to normalize the cost
    order :: int - the order with which to take the differences
        of controls - i.e. de/dt = order 1, de^2/dt^2 = order 2, etc.
    """
    name = "control_variation"
    requires_step_evaluation = False

    def __init__(self, control_count, control_step_count,
                 max_control_norms, cost_multiplier=1.,
                 order=1):
        """
        See class definition for arguments not listed here.
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_count = control_count
        self.control_step_count = control_step_count
        self.max_control_norms = max_control_norms
        self.normalization_constant = control_count * (control_step_count - order)
        self.order = order


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

        # Penalize the difference in variations from the value of a control
        # at one step to the next step.
        diffs = anp.diff(normalized_controls, axis=0, n=self.order)
        diffs_total = anp.sum(anp.square(anp.abs(diffs)))
        diffs_total_normalized = anp.divide(diffs_total, self.normalization_constant)

        return self.cost_multiplier * diffs_total_normalized


def _test():
    """
    Run test on the module.
    """
    controls = np.array(((1+2j, 7+8j,), (3+4j, 9+10j,), (11+12j, 5+6j,),))
    control_count = controls.shape[1]
    control_step_count = controls.shape[0]
    max_control_norms = np.array((np.sqrt(265), np.sqrt(181),))
    
    cvo1 = ControlVariation(control_count, control_step_count,
                            max_control_norms,
                            order=1)
    cost = cvo1.cost(controls, None, None)
    expected_cost = 0.18355050557698324

    assert(np.allclose(cost, expected_cost))

    cvo2 = ControlVariation(control_count,
                            control_step_count,
                            max_control_norms,
                            order=2)
    cost = cvo2.cost(controls, None, None)
    expected_cost = 0.33474408422808305
    
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _test()
            
