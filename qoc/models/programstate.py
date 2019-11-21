"""
programstate.py - This module defines classes to encapsulate data fields
necessary to execute qoc programs.
"""

from filelock import FileLock
import numpy as np

from qoc.models.programtype import ProgramType

class ProgramState(object):
    """
    This class encapsulates data fields that are used
    by most programs.

    Fields:
    control_eval_count
    control_eval_times
    cost_eval_step
    costs
    dt
    evolution_time
    final_system_eval_step
    hamiltonian
    interpolation_policy
    program_type
    save_file_lock_path
    save_file_path
    step_cost_indices
    step_costs
    system_eval_count
    """
    def __init__(self, control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian, interpolation_policy,
                 program_type,
                 save_file_path, system_eval_count):
        """
        See class fields for arguments not listed here.
        """
        self.control_eval_count = control_eval_count
        self.control_eval_times = np.linspace(0, evolution_time, control_eval_count)
        self.cost_eval_step = cost_eval_step
        self.costs = costs
        self.dt = evolution_time / (system_eval_count - 1)
        self.evolution_time = evolution_time
        self.final_system_eval_step = system_eval_count - 1
        self.hamiltonian = hamiltonian
        self.interpolation_policy = interpolation_policy
        self.program_type = program_type
        self.save_file_lock_path = "{}.lock".format(save_file_path)
        self.save_file_path = save_file_path
        step_cost_indices = list()
        step_costs = list()
        for i, cost in enumerate(costs):
            if cost.requires_step_evaluation:
                step_costs.append(cost)
                step_cost_indices.append(i)
        #ENDFOR
        self.step_cost_indices = step_cost_indices
        self.step_costs = step_costs
        self.system_eval_count = system_eval_count


class GrapeState(ProgramState):
    """
    This class encapsulates data fields that are used
    by most grape programs.
    
    Fields:
    complex_controls
    control_count
    control_eval_count
    control_eval_times
    controls_shape
    cost_eval_step
    costs
    dt
    evolution_time
    final_iteration
    final_system_eval_step
    hamiltonian
    impose_control_conditions
    initial_controls
    interpolation_policy
    iteration_count
    log_iteration_step
    max_control_norms
    min_error
    optimizer
    program_type
    save_file_lock_path
    save_file_path
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    system_eval_count
    """

    def __init__(self, complex_controls,
                 control_count,
                 control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian,
                 impose_control_conditions,
                 initial_controls,
                 interpolation_policy, iteration_count,
                 log_iteration_step, max_control_norms,
                 min_error, optimizer,
                 save_file_path, save_iteration_step,
                 system_eval_count,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__(control_eval_count, cost_eval_step, costs,
                         evolution_time, hamiltonian, interpolation_policy,
                         ProgramType.GRAPE,
                         save_file_path, system_eval_count,)
        self.complex_controls = complex_controls
        self.control_count = control_count
        self.controls_shape = (control_eval_count, control_count)
        self.final_iteration = iteration_count - 1
        self.impose_control_conditions = impose_control_conditions
        self.initial_controls = initial_controls
        self.iteration_count = iteration_count
        self.log_iteration_step = log_iteration_step
        self.max_control_norms = max_control_norms
        self.min_error = min_error
        self.optimizer = optimizer
        self.save_iteration_step = save_iteration_step
        self.should_log = log_iteration_step != 0
        self.should_save = ((save_iteration_step != 0)
                            and (not (save_file_path is None)))
