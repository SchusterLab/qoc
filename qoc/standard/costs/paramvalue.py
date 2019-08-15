"""
paramvalue.py - a module to define a cost on parameter values
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ParamValue(Cost):
    """
    a cost to penalize high parameter values
    Fields:
    cost_multiplier :: float - the weight factor for this cost
    max_param_norms :: numpy.ndarray - the maximum absolute values for
        each param, the array's shape should be (param_count,)
    name :: str - a unique identifier for this cost
    nomalization_constant :: int - param_count * pulse_step_count
    param_count :: int - the number of parameters at each
        time step
    pulse_step_count :: int - the number of time steps
    """
    name = "param_variation"
    requires_step_evaluation = False

    def __init__(self, max_param_norms,
                 param_count, pulse_step_count, cost_multiplier=1.,):
        """
        See class fields for argument information.
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_param_norms = max_param_norms
        self.normalization_constant = param_count * pulse_step_count
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
        # Normalize the parameters.
        normalized_params = anp.divide(params, self.max_param_norms)

        # Penalize the sum of the square of the absolute value of the parameters.
        cost = anp.sum(anp.square(anp.abs(normalized_params)))
        cost_normalized = anp.divide(cost, self.normalization_constant)
        
        return self.cost_multiplier * cost_normalized


def _test():
    """
    Run test on the module.
    """
    max_param_norms = np.array((np.sqrt(181), np.sqrt(265),))
    params = np.array(((1+2j, 3-4j), (5-6j, 7+8j), (9+10j, 11+12j),))
    param_count = params.shape[1]
    pulse_step_count = params.shape[0]
    pv = ParamValue(max_param_norms, param_count,
                    pulse_step_count)
    
    cost = pv.cost(params, None, None)
    expected_cost = 0.48089926682650547
    
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _test()
