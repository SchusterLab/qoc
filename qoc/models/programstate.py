"""
programstate.py - This module defines classes that encapsulate the necessary
information to execute programs.
"""

class ProgramState(object):
    """
    Fields:
    costs :: iterable(qoc.models.Cost) - the cost functions that
        define the cost model for evolution
    dt :: float - the time step used for evolution, it is the time
        inbetween system steps
    final_control_step :: int - the ultimate index into the control array
    final_system_step :: int - the last time step
    hamiltonian :: (controls :: ndarray, time :: float)
                    -> hamiltonian :: ndarray
        - an autograd compatible function that returns the system
          hamiltonian for the specified control parameters and time
    interpolation_policy :: qoc.InterpolationPolicy - defines how
        the control parameters should be interpolated between
        control steps
    operation_policy :: qoc.models.OperationPolicy - defines how
        computations should be performed, e.g. CPU, GPU, sparse, etc.
    step_cost_indices :: iterable(int)- a list of the indices in the costs
        array which are step costs
    step_costs :: iterable(qoc.models.Cost) - the cost functions that
        define the cost model for evolution that should be evaluated
        at every time step
    system_step_multiplier :: int - this value times `control_step_count`
        determines how many time steps are used in evolution
    """
    def __init__(self, control_step_count, costs, evolution_time,
                 hamiltonian,
                 interpolation_policy,
                 operation_policy, system_step_multiplier):
        """
        See class definition for arguments not listed here.

        Args:
        control_step_count :: int - the number of time steps at which the
            evolution time should be initially split into and the number
            of control parameter updates
        evolution_time :: float - the time over which the system will evolve
        """
        self.costs = costs
        system_step_count = control_step_count * system_step_multiplier
        self.dt = evolution_time / system_step_count
        self.final_control_step = control_step_count - 1
        self.final_system_step = system_step_count - 1
        self.hamiltonian = hamiltonian
        self.interpolation_policy = interpolation_policy
        self.operation_policy = operation_policy
        self.step_cost_indices = list()
        self.step_costs = list()
        for i, cost in enumerate(costs):
            if cost.requires_step_evaluation:
                self.step_costs.append(cost)
                self.step_cost_indices.append(i)
        #ENDFOR
        self.system_step_multiplier = system_step_multiplier


class GrapeState(ProgramState):
    """
    This class encapsulates the necessary information to execute
    a grape program.
    
    Fields:
    complex_controls
    control_count
    costs
    dt
    final_control_step
    final_system_step
    hamiltonian
    initial_controls
    interpolation_policy
    iteration_count
    log_iteration_step
    max_control_norms
    operation_policy
    optimizer
    save_file_path
    save_iteration_step
    step_cost_indices
    step_costs
    system_step_multiplier
    """

    def __init__(self, complex_controls,
                 control_count,
                 control_step_count, costs, evolution_time,
                 hamiltonian, initial_controls,
                 interpolation_policy, iteration_count,
                 log_iteration_step, max_control_norms,
                 operation_policy, optimizer,
                 save_file_path, save_iteration_step,
                 system_step_multiplier,):
        """
        See class definition for arguments not listed here.

        Args:
        control_count
        control_step_count
        evolution_time
        """
        super().__init__(control_step_count, costs, evolution_time,
                         hamiltonian, interpolation_policy, operation_policy,
                         system_step_multiplier)
        self.complex_controls = complex_controls
        self.control_count = control_count
        self.initial_controls = initial_controls
        self.iteration_count = iteration_count
        self.log_iteration_step = log_iteration_step
        self.max_control_norms = max_control_norms
        self.optimizer = optimizer
        self.save_file_path = save_file_path
        self.save_iteration_step = save_iteration_step
        
        
