"""
forbidstates.py - a module to encapsulate the forbidden states cost function
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost
from qoc.standard.functions import conjugate_transpose

class ForbidStates(Cost):
    """
    This class encapsulates a cost function that penalizes
    the occupation of forbidden states.

    Fields:
    cost_multiplier :: float - the wieght factor for this cost
    forbidden_states_dagger :: ndarray - the conjugate transpose of
        the forbidden states
    name :: str - a unique identifier for this cost
    normalization_constant :: int - used to normalize the cost
    requires_step_evaluation :: bool - True if the cost needs
        to be computed at each optimization time step, False
        if it should be computed only at the final optimization
        time step
    state_normalization_constants :: ndarray - the number of states
        that each evolving state is forbidden from
    """
    name = "forbid_states"
    requires_step_evaluation = True


    def __init__(self, forbidden_states, system_step_count, cost_multiplier=1.):
        """
        See class definition for arguments not listed here.

        Args:
        forbidden_states :: ndarray - an array where each entry
            in the first axis is an array of states that the corresponding
            evolving state is forbidden from, that is, each evolving
            state has its own list of forbidden states
        system_step_count :: int - the number of system steps in the evolution
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.forbidden_states_dagger = conjugate_transpose(forbidden_states)
        state_count = forbidden_states.shape[0]
        self.normalization_constant = state_count * system_step_count
        self.state_normalization_constants = np.array([state_forbidden_states.shape[0]
                                                       for state_forbidden_states
                                                       in forbidden_states])


    def cost(self, controls, states, system_step):
        """
        Args:
        controls :: ndarray - the control parameters for all time steps
        states :: ndarray - an array of the initial states evolved to
            the current time step
        system_step :: int - the system time step

        Returns:
        cost :: float - the penalty
        """
        cost = 0
        # Compute the fidelity for each evolution state and its forbidden states.
        for i, state_forbidden_states_dagger in enumerate(self.forbidden_states_dagger):
            state = states[i]
            state_cost = 0
            for forbidden_state_dagger in state_forbidden_states_dagger:
                inner_product = anp.matmul(forbidden_state_dagger, state)[0, 0]
                state_cost = state_cost + anp.square(anp.abs(inner_product))
            #ENDFOR
            cost = cost + anp.divide(state_cost, self.state_normalization_constants[i])
        #ENDFOR
        
        # Normalize the cost for the number of evolving states
        # and the number of time evolution steps.
        cost = (cost / self.normalization_constant)
        
        return self.cost_multiplier * cost


def _test():
    """
    Run tests on the module.
    """
    system_step_count = 10
    state0 = np.array([[1], [0]])
    forbid0_0 = np.array([[1], [0]])
    forbid0_1 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    state1 = np.array([[0], [1]])
    forbid1_0 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    forbid1_1 = np.divide(np.array([[1j], [1j]]), np.sqrt(2))
    states = np.stack((state0, state1,))
    forbidden_states0 = np.stack((forbid0_0, forbid0_1,))
    forbidden_states1 = np.stack((forbid1_0, forbid1_1,))
    forbidden_states = np.stack((forbidden_states0, forbidden_states1,))
    fs = ForbidStates(forbidden_states, system_step_count)
    
    cost = fs.cost(None, states, None)
    expected_cost = np.divide(5, 80)
    assert(np.allclose(cost, expected_cost,))


if __name__ == "__main__":
    _test()
