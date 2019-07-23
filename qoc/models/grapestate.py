"""
grapestate.py - a module for classes to encapsulate the state of a
GRAPE optimization
"""

class GrapeStateDiscrete(object):
    """
    The class to encapsulate the necessary information to perform a
    discrete GRAPE optimization.
    Fields:
    system_hamiltonian :: (time :: float, params :: np.ndarray)
                          -> hamiltonian :: np.ndarray - an autograd compatible
                          (https://github.com/HIPS/autograd) function that
                          returns the system hamiltonian given the evolution
                          time and control parameters
    parameter_count :: int - the number of control parameters required at each
                             optimization time step
    initial_states :: [np.ndarray] - the states to evolve
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    pulse_time :: float - the duration of the pulse
    pulse_step_count :: int - the number of time steps at which the pulse should
                              be optimized
    system_step_count :: int - the number of time steps at which the system should
                               evolve, control parameters will be interpolated at
                               these time steps if a value different from pulse_steps
                               is specified
    magnus_order :: qoc.MagnusOrder - the accuracy of the magnus expansion
    operation_type :: qoc.OperationType - how computations should be performed,
                                          e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.models.Optimizer - an instance of an optimizer to perform
                                        gradient-based optmization
    """

    def __init__(self, system_hamiltonian, parameter_count, initial_states,
                 costs, pulse_time, pulse_step_count, system_step_count,
                 magnus_order, operation_type, optimizer):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.system_hamiltonian = system_hamiltonian
        self.parameter_count = parameter_count
        self.initial_statse = initial_states
        self.costs = costs
        self.pulse_time = pulse_time
        self.pulse_step_count = pulse_step_count
        self.system_step_count = system_step_count
        self.magnus_order = magnus_order
        self.operation_type = operation_type
        self.optimizer = optimizer
