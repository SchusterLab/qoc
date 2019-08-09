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
    max_param_amplitudes :: numpy.ndarray - the maximum values for
        each param, the array's shape should be (param_count,)
    name :: str - a unique identifier for this cost
    param_count :: int - the number of parameters at each
        time step
    pulse_step_count :: int - the number of time steps
    """
    name = "param_variation"
    requires_step_evaluation = False

    def __init__(self, max_param_amplitudes,
                 param_count, pulse_step_count, cost_multiplier=1.,):
        """
        See class fields for argument information.
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_param_amplitudes = max_param_amplitudes
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
        max_param_amplitudes = self.max_param_amplitudes
        param_count = self.param_count

        # Normalize the parameters.
        normalized_params = anp.zeros_like(params)
        for i in range(param_count):
            normalized_params[:,i] = (params[:,i]
                                       / max_param_amplitudes[i])

        # Penalize the sum of the square of the absolute value of the parameters.
        cost = anp.sum(anp.square(anp.abs(normalized_params)))
        cost_normalized = anp.divide(anp.divide(cost, param_count),
                                     self.pulse_step_count)
        return self.cost_multiplier * cost_normalized


def _test():
    """
    Run test on the module.
    """
    max_param_amplitudes = np.array([1, 2])
    params = np.array([[1, 2], [.5, 1.5], [0, 1]])
    param_count = params.shape[1]
    pulse_step_count = params.shape[0]
    
    pv = ParamValue(max_param_amplitudes, param_count,
                    pulse_step_count)
    cost = pv.cost(params, None, None)
    expected_cost = 0.5104166666666666
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _test()
            
