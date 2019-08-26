"""
programstate.py - This module defines a superclass
for classes that encapsulate the necessary information
to execute a program.
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
    step_costs :: iterable(qoc.models.Cost) - the cost functions that
        define the cost model for evolution that should be evaluated
        at every time step
    step_cost_indices :: iterable(int)- a list of the indices in the costs
        array which are step costs
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
        self.step_costs = list()
        self.step_cost_indices = list()
        for i, cost in enumerate(costs):
            if cost.requires_step_evaluation:
                self.step_costs.append(cost)
                self.step_cost_indices.append(i)
        #ENDFOR
        self.system_step_multiplier = system_step_multiplier
