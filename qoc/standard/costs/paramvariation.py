"""
paramvariation.py - a module for defining cost functions related
to variations in control amplitudes
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ParamVariation(Cost):
    """
    a cost to penalize variations of control parameters
    Fields:
    cost_multiplier :: float - the weight factor for this cost
    max_param_norms :: numpy.ndarray - the maximum absolute values for
        each param, the array's shape should be (param_count,)
    name :: str - a unique identifier for this cost
    order :: int - the order with which to take the differences
        of parameters - i.e. de/dt = order 1, de^2/dt^2 = order 2, etc.
    param_count :: int - the number of parameters at each
        time step
    pulse_step_count :: int - the number of time steps,
        we require pulse_step_count >= order + 1
    """
    name = "param_variation"
    requires_step_evaluation = False

    def __init__(self, max_param_norms,
                 param_count, pulse_step_count, cost_multiplier=1.,
                 order=1):
        """
        See class fields for argument information.
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_param_norms = max_param_norms
        self.order = order
        self.param_count = param_count
        self.pulse_step_count = pulse_step_count

    
    def cost(self, params, states, step):
        """
        Args:
        params :: numpy.ndarray - the control parameters for all time steps
        states :: numpy.ndarray - an arry of the initial states evolved to
            the current time step
        step :: int - the pulse time step
        Returns:
        cost :: float - the penalty
        """
        # Heap -> Stack.
        max_param_norms = self.max_param_norms
        order = self.order
        param_count = self.param_count

        # Normalize the parameters.
        normalized_params = anp.divide(params, max_param_norms)

        # Penalize the difference in variations from the value of a parameter
        # at one step to the next step.
        diffs = anp.diff(normalized_params, axis=0, n=order)
        diffs_total = anp.sum(anp.square(anp.abs(diffs)))
        diffs_total_normalized = anp.divide(diffs_total,
                                            param_count * (self.pulse_step_count - order))

        return self.cost_multiplier * diffs_total_normalized


def _test():
    """
    Run test on the module.
    """
    max_param_norms = np.array((np.sqrt(265), np.sqrt(181),))
    params = np.array(((1+2j, 7+8j,), (3+4j, 9+10j,), (11+12j, 5+6j,),))
    param_count = params.shape[1]
    pulse_step_count = params.shape[0]
    
    pvo1 = ParamVariation(max_param_norms, param_count,
                          pulse_step_count, order=1)
    cost = pvo1.cost(params, None, None)
    expected_cost = 0.18355050557698324

    assert(np.allclose(cost, expected_cost))

    pvo2 = ParamVariation(max_param_norms, param_count,
                          pulse_step_count, order=2)
    cost = pvo2.cost(params, None, None)
    expected_cost = 0.33474408422808305
    
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _test()
            
