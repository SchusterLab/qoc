"""
schroedingermodels.py - This module defines classes that encapsulate 
data fiels used by Schroedinger equation programs.
"""

import os

from filelock import FileLock, Timeout
import h5py
import numpy as np

from qoc.models.programstate import (GrapeState,)


class GrapeSchroedingerDiscreteState(GrapeState):
    """
    This class encapsulates the data fields used by the
    qoc.core.schroedingerdiscrete.grap_schroedinger_discrete
    program.

    Fields:
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
    impose_control_conditions
    initial_controls
    initial_states
    iteration_count
    log_iteration_step
    max_control_norms
    method
    min_error
    optimizer
    program_type
    save_file_lock_path
    save_file_path
    save_intermediate_states_
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    gradients_method
    expm_method
    """
    method = "grape_schroedinger_discrete"

    def __init__(self, H_s, H_controls,
                 control_eval_count,  costs,
                 evolution_time,
                 impose_control_conditions,
                 initial_controls,
                 initial_states, iteration_count,
                 log_iteration_step, max_control_norms,
                  min_error, optimizer,
                 save_file_path, save_intermediate_states_,
                 save_iteration_step, gradients_method, expm_method
                 ):
        """
        See class fields for arguments not listed here.
        """
        control_count = len(H_controls)
        self.control_count = control_count
        complex_controls = False
        cost_eval_step = 1
        hamiltonian = None
        system_eval_count = control_eval_count
        interpolation_policy = None
        super().__init__(complex_controls, control_count,
                         control_eval_count, cost_eval_step, costs,
                         evolution_time, hamiltonian,
                         impose_control_conditions,
                         initial_controls,
                         interpolation_policy, iteration_count,
                         log_iteration_step, max_control_norms,
                         min_error, optimizer,
                         save_file_path, save_iteration_step,
                         system_eval_count,)
        self.hilbert_size = initial_states[0].shape[0]
        self.initial_states = initial_states
        self.save_intermediate_states_ = (self.should_save
                                          and save_intermediate_states_)
        self.H_s = H_s
        self.H_controls = H_controls
        self.gradients_method = gradients_method
        self.expm_method = expm_method

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
        # Don't log if the iteration number is invalid.
        if iteration > self.final_iteration:
            return
        
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
            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "a") as save_file:
                        save_file["controls"][save_step,] = controls
                        save_file["error"][save_step,] = error
                        save_file["final_states"][save_step,] = final_states
                        save_file["grads"][save_step,] = grads
                    #ENDWITH
                #ENDWITH
            except Timeout:
                print("Timeout while locking {} to save after iteration {}."
                      "".format(self.save_file_lock_path, iteration))


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

            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "w") as save_file:
                        save_file["H_s"] = self.H_s
                        save_file["H_controls"] = self.H_controls
                        save_file["gradients_mode"] = self.gradients_method
                        save_file["control_eval_count"] = self.control_eval_count
                        save_file["controls"] = np.zeros((save_count, self.control_eval_count,
                                                          self.control_count,),
                                                         dtype=self.initial_controls.dtype)
                        save_file["cost_eval_step"] = self.cost_eval_step
                        save_file["cost_names"] = np.array([np.string_("{}".format(cost))
                                                            for cost in self.costs])
                        save_file["error"] = np.repeat(np.finfo(np.float64).max, save_count)
                        save_file["evolution_time"]= self.evolution_time
                        save_file["final_states"] = np.zeros((save_count, state_count,
                                                              self.hilbert_size, 1),
                                                             dtype=np.complex128)
                        save_file["grads"] = np.zeros((save_count, self.control_eval_count,
                                                       self.control_count), dtype=self.initial_controls.dtype)
                        save_file["initial_controls"] = self.initial_controls
                        save_file["initial_states"] = self.initial_states
                        if self.save_intermediate_states_:
                            save_file["intermediate_states"] = np.zeros((save_count,
                                                                         self.system_eval_count,
                                                                         *self.initial_states.shape),
                                                                        dtype=np.complex128)
                        save_file["iteration_count"] = self.iteration_count
                        save_file["max_control_norms"] = self.max_control_norms
                        save_file["method"] = self.method
                        save_file["optimizer"] = "{}".format(self.optimizer)
                        save_file["program_type"] = self.program_type.value
                    #ENDWITH
                #ENDWITH
            except Timeout:
                print("Timeout while locking {}, could not perform initial save."
                      "".format(self.save_file_lock_path))
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")

    
    def save_intermediate_states(self, iteration,
                                 states, system_eval_step,):
        """
        Save intermediate states to the save file.
        """
        # Don't log if the iteration number is invalid.
        if iteration > self.final_iteration:
            return

        # Determine decision parameters.
        is_final_iteration = iteration == self.final_iteration
        
        if (self.should_save
            and ((np.mod(iteration, self.save_iteration_step) == 0)
                 or is_final_iteration)):
            save_step, _ = np.divmod(iteration, self.save_iteration_step)
            try:
                with FileLock(self.save_file_lock_path):
                    with h5py.File(self.save_file_path, "a") as save_file:
                        save_file["intermediate_states"][save_step, system_eval_step, :, :, :] = states.astype(np.complex128)
            except Timeout:
                print("Timeout while locking {}, could not save intermediate states on iteration {} and "
                      "system_eval_step {}."
                      "".format(self.save_file_lock_path, iteration, system_eval_step))
        #ENDIF


class GrapeSchroedingerResult(object):
    """
    This class encapsulates the result of the
    qoc.core.lindbladdiscrete.grape_schroedinger_discrete
    program.

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
