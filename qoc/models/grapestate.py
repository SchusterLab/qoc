"""
grapestate.py - a module for classes to encapsulate the state of a
GRAPE optimization
"""

class GrapeStateDiscrete(object):
    """
    a class to encapsulate the necessary information to perform a
    discrete GRAPE optimization.
    Fields:
    system_hamiltonian :: (time :: float, params :: numpy.ndarray)
                          -> hamiltonian :: numpy.ndarray
      - an autograd compatible (https://github.com/HIPS/autograd) function that
        returns the system hamiltonian given the evolution time
        and control parameters
    parameter_count :: int - the number of control parameters required at each
         optimization time step
    initial_states :: [numpy.ndarray] - the states to evolve
    final_costs :: [qoc.models.Cost] - the cost functions to guide optimization
        that are evaluated at the end of the evolution
    step_costs :: [qoc.models.Cost] - the cost functions to guide optimization
        that are evaluated at each step of optimization
    iteration_count :: int - the number of iterations to optimize for
    pulse_time :: float - the duration of the control pulse
    pulse_step_count :: int - the number of time steps at which the pulse
        should be optimized
    system_step_multiplier :: int - the multiple of pulse_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps
    magnus_method :: qoc.MagnusMethod - the method to use for the magnus
        expansion
    operation_type :: qoc.OperationType - how computations should be performed,
        e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.models.Optimizer - an instance of an optimizer to perform
        gradient-based optimization
    """
    def __init__(self, system_hamiltonian, parameter_count, initial_states,
                 final_costs, step_costs, pulse_time, pulse_step_count,
                 system_step_multiplier, magnus_method, operation_type,
                 optimizer):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.system_hamiltonian = system_hamiltonian
        self.parameter_count = parameter_count
        self.initial_statse = initial_states
        self.final_costs = final_costs
        self.step_costs = step_costs
        self.pulse_time = pulse_time
        self.pulse_step_count = pulse_step_count
        self.system_step_multiplier = system_step_multiplier
        self.magnus_method= magnus_method
        self.operation_type = operation_type
        self.optimizer = optimizer


class GrapeResult(object):
    """
    a class to encapsulate the results of a GRAPE optimization
    Fields:
    final_states :: [numpy.ndarray] - the resultant, evolved final states
    parameters :: numpy.ndarray - the resultant, evolved final states
    """
    def __init__(self, final_states, parameters):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.final_states = final_states
        self.parameters = parameters

        
class EvolveResult(object):
    """
    a class to encapsulate the results of an evolution
    Fields:
    final_states :: [numpy.ndarray] - the resultant, evolved final states
    """
    def __init__(self, final_states):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.final_states = final_states
