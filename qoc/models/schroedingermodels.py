"""
schroedingermodels.py - This module defines classes that encapsulate 
data fiels used by Schroedinger equation programs.
"""

import os

import h5py
import numpy as np

from qoc.models.programstate import (ProgramState, GrapeState,)

class EvolveSchroedingerDiscreteState(ProgramState):
    """
    This class encapsulates data fields that are used by the
    qoc.core.schroedingerdiscrete.evolve_schroedinger_discrete
    program.
    
    Fields:
    control_eval_count
    control_eval_times
    cost_eval_step
    costs
    dt
    evolution_time
    final_system_eval_step
    hamiltonian
    initial_states
    interpolation_policy
    magnus_policy
    method
    save_file_path
    save_intermediate_states
    step_cost_indices
    step_costs
    system_eval_count
    """
    method = "evolve_schroedinger_discrete"
    
    def __init__(self,control_eval_count,
                 cost_eval_step, costs,
                 evolution_time, hamiltonian, initial_states,
                 interpolation_policy,
                 magnus_policy, save_file_path,
                 save_intermediate_states,
                 system_eval_count,):
        """
        See class definition for arguments not listed here.
        """
        super().__init__(control_eval_count, cost_eval_step, costs,
                         evolution_time, hamiltonian, interpolation_policy,
                         save_file_path, system_eval_count,)
        self.initial_states = initial_states
        self.magnus_policy = magnus_policy
        self.save_intermediate_states = (save_intermediate_states
                                         and (save_file_path is not None))


    def save_initial(self):
        """
        Perform the initial save.
        """
        if self.save_file_path is not None:
            print("QOC is saving this evolution to {}."
                  "".format(self.save_file_path))

            with h5py.File(self.save_file_path, "w") as save_file:
                save_file["controls"] = self.controls
                save_file["cost_eval_step"] = self.cost_eval_step
                save_file["costs"] = np.array(["{}".format(cost)
                                               for cost in self.costs])
                save_file["evolution_time"] = self.evolution_time
                save_file["initial_states"] = self.initial_states
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                save_file["intermediate_states"] = np.zeros((self.system_eval_count,
                                                             *self.initial_states.shape),
                                                            dtype=np.complex128)
                save_file["magnus_policy"] = "{}".format(self.magnus_policy)
                save_file["method"] = self.method
                save_file["system_eval_count"] = self.system_eval_count

    
    def save_intermediate_states(self, states, system_eval_step):
        """
        Save intermediate states to the save file.
        """
        with h5py.File(self.save_file_path, "a") as save_file:
            save_file["intermediate_states"][system_eval_step] = states


class EvolveSchroedingerResult(object):
    """
    This class encapsulates the result of the
    qoc.core.schroedingerdiscrete.evolve_schroedinger_discrete
    program.
    
    Fields:
    error
    final_states
    """

    def __init__(self, error=None,
                 final_states=None,):
        """
        See the class fields for arguments not listed here.
        """
        super().__init__()
        self.error = error
        self.final_states = final_states


class GrapeSchroedingerDiscreteState(GrapeState):
    """
    This class encapsulates the data fields used by the
    qoc.core.schroedingerdiscrete.grap_schroedinger_discrete
    program.

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
    hilbert_size
    initial_controls
    initial_states
    interpolation_policy
    iteration_count
    log_iteration_step
    max_control_norms
    magnus_policy
    method
    min_error
    optimizer
    save_file_path
    save_intermediate_states
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    system_eval_count
    """
    method = "grape_schroedinger_discrete"
    save_intermediate_states = False

    def __init__(self, complex_controls, control_count,
                 control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian, initial_controls,
                 initial_states, interpolation_policy, iteration_count,
                 log_iteration_step, max_control_norms,
                 magnus_policy, min_error, optimizer,
                 save_file_path, save_iteration_step,
                 system_eval_count,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__(complex_controls, control_count,
                         control_eval_count, cost_eval_step, costs,
                         evolution_time, hamiltonian, initial_controls,
                         interpolation_policy, iteration_count,
                         log_iteration_step, max_control_norms,
                         min_error, optimizer,
                         save_file_path, save_iteration_step,
                         system_eval_count,)
        self.hilbert_size = initial_states[0].shape[0]
        self.initial_states = initial_states
        self.magnus_policy = magnus_policy


    def log_and_save(self, controls, error, final_states, grads, iteration,):
        """
        If necessary, log to stdout and save to the save file.

        Arguments:
        controls :: ndarray - the optimization parameters
        error :: ndarray - the total error at the last time step
            of evolution
        final_states :: ndarray - the states at the last time step
            of evolution
        grads :: ndarray - the current gradients of the cost function
            with resepct to controls
        iteration :: int - the optimization iteration
        

        Returns: none
        """
        # Determine decision parameters.
        is_final_iteration = iteration == self.final_iteration
        
        if (self.should_log
            and ((np.mod(iteration, self.log_iteration_step) == 0)
                 or is_final_iteration)):
            grads_norm = np.linalg.norm(grads)
            print("{:^6d} | {:^1.8e} | {:^1.8e}"
                  "".format(iteration, error,
                            grads_norm))

        if (self.should_save
            and ((np.mod(iteration, self.save_iteration_step) == 0)
                 or is_final_iteration)):
            save_step, _ = np.divmod(iteration, self.save_iteration_step)
            with h5py.File(self.save_file_path, "a") as save_file:
                save_file["controls"][save_step,] = controls
                save_file["error"][save_step,] = error
                save_file["final_states"][save_step,] = states
                save_file["grads"][save_step,] = grads


    def log_and_save_initial(self):
        """
        Perform the initial log and save.
        """
        if self.should_save:
            # Notify the user where the file is being saved.
            print("QOC is saving this optimization run to {}."
                  "".format(self.save_file_path))

            save_count, save_count_remainder = np.divmod(self.iteration_count,
                                                         self.save_iteration_step)
            state_count = len(self.initial_states)
            # If the final iteration doesn't fall on a save step, add a save step.
            if save_count_remainder != 0:
                save_count += 1

            with h5py.File(self.save_file_path, "w") as save_file:
                save_file["complex_controls"] = self.complex_controls
                save_file["control_count"] = self.control_count
                save_file["control_eval_count"] = self.control_eval_count
                save_file["controls"] = np.zeros((save_count, self.control_step_count,
                                                  self.control_count,),
                                                 dtype=self.initial_controls.dtype)
                save_file["cost_eval_step"] = self.cost_eval_step
                save_file["cost_names"] = np.array([np.string_("{}".format(cost))
                                                    for cost in self.costs])
                save_file["error"] = np.zeros((save_count),
                                              dtype=np.float64)
                save_file["evolution_time"]= self.evolution_time
                save_file["grads"] = np.zeros((save_count, self.control_eval_count,
                                               self.control_count), dtype=self.initial_controls.dtype)
                save_file["initial_controls"] = self.initial_controls
                save_file["initial_states"] = self.initial_states
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                save_file["iteration_count"] = self.iteration_count
                save_file["magnus_policy"] = "{}".format(self.magnus_policy)
                save_file["max_control_norms"] = self.max_control_norms
                save_file["method"] = self.method
                save_file["optimizer"] = "{}".format(self.optimizer)
                save_file["final_states"] = np.zeros((save_count, state_count,
                                                      self.hilbert_size, 1),
                                                     dtype=np.complex128)
                save_file["system_eval_count"] = self.system_eval_count
            #ENDWITH
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")


class GrapeSchroedingerResult(object):
    """
    This class encapsulates useful information about a
    grape optimization under the Schroedinger equation.

    Fields:
    best_controls
    best_error
    best_final_states
    best_iteration
    """
    def __init__(self, best_controls=None,
                 best_error=np.finfo(np.float64).max,
                 best_final_states=None,
                 best_iteration=None,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__()
        self.best_controls = best_controls
        self.best_error = best_error
        self.best_final_states = best_final_states
        self.best_iteration = best_iteration
