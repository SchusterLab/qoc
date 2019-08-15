"""
targetinfidelitytime.py - a module for defining the target fidelity cost function
evaluated at each time step
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class TargetInfidelityTime(Cost):
    """a class to encapsulate the target infidelity cost function
    Fields:
    cost_multiplier :: float - the wieght factor for this cost
    dcost_dparams :: (params :: numpy.ndarray, states :: numpy.ndarray, step :: int)
                      -> dcost_dparams :: numpy.ndarray
        - the gradient of the cost function with respect to the parameters
    dcost_dstates :: (params :: numpy.ndarray, states :: numpy.ndarray, step :: int)
                      -> dcost_dstates :: numpy.ndarray
        - the gradient of the cost function with respect to the states
    name :: str - a unique identifier for this cost
    requires_step_evaluation :: bool - True if the cost needs
        to be computed at each optimization time step, False
        if it should be computed only at the final optimization
        time step
    state_count :: float - value used to compute
        the cost averaged over the states
    step_count :: int - the number of steps in the evolution
    target_states_dagger :: numpy.ndarray - the hermitian conjugate of
        the target states
    """
    name = "target_infidelity"
    requires_step_evaluation = True


    def __init__(self, step_count, target_states, cost_multiplier=1.):
        """
        See class definition for parameter specification.
        target_states :: numpy.ndarray - an array of states
            that correspond to the target state for each of the initial states
            used in optimization
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.state_count = target_states.shape[0]
        self.step_count = step_count
        self.target_states_dagger = conjugate_transpose(anp.stack(target_states))
        # This cost function does not make use of parameter penalties.
        self.dcost_dparams = (lambda params, states, step:
                              np.zeros_like(params))


    def cost(self, params, states, step):
        """
        Args:
        params :: numpy.ndarray - the control parameters for all time steps
        states :: numpy.ndarray - an array of the states evolved to
            the current time step
        step :: int - the pulse time step
        Returns:
        cost :: float - the penalty
        """
        fidelity = anp.sum(anp.square(anp.abs(anp.matmul(self.target_states_dagger,
                                                         states)[:,0,0])), axis=0)
        fidelity_normalized = anp.divide(fidelity, self.state_count)
        infidelity = 1 - fidelity_normalized
        infidelity_normalized = anp.divide(infidelity, self.step_count)
        return self.cost_multiplier * infidelity_normalized


def _tests():
    """
    Run test on the module.
    """
    step_count = 10
    state0 = np.array([[0], [1]])
    target0 = np.array([[1], [0]])
    states = np.stack((state0,), axis=0)
    targets = np.stack((target0,), axis=0)
    ti = TargetInfidelityTime(step_count, targets)
    cost = ti.cost(None, states, None)
    expected_cost = 0.1
    assert(np.allclose(cost, expected_cost))

    ti = TargetInfidelityTime(step_count, states)
    cost = ti.cost(None, states, None)
    expected_cost = 0
    assert(np.allclose(cost, expected_cost))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    target0 = np.array([[1j], [0]])
    target1 = np.array([[1], [0]])
    states = np.stack((state0, state1,), axis=0)
    targets = np.stack((target0, target1,), axis=0)
    ti = TargetInfidelityTime(step_count, targets)
    cost = ti.cost(None, states, None)
    expected_cost = 0.025
    assert(np.allclose(cost, expected_cost))


if __name__ == "__main__":
    _tests()
