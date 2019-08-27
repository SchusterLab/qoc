"""
schroedingermodels.py - This module defines classes that encapsulate necessary
information to run programs involving the Schroedinger equation.
"""

import os

import h5py
import numpy as np

from qoc.models.programstate import (ProgramState, GrapeState,)

class EvolveSchroedingerDiscreteState(ProgramState):
    """
    This class encapsulates the necessary information to run
    qoc.core.schroedingerdiscrete.evolve_schroedinger_discrete.
    
    Fields:
    costs
    dt
    final_control_step
    final_system_step
    hamiltonian
    initial_states
    interpolation_policy
    magnus_policy
    operation_policy
    step_costs
    step_cost_indices
    system_step_multiplier
    """
    
    def __init__(self, control_step_count, costs, evolution_time,
                 hamiltonian, initial_states, interpolation_policy,
                 magnus_policy, operation_policy, system_step_multiplier):
        """
        See class definition for arguments not listed here.

        Args:
        control_step_count
        evolution_time
        """
        super().__init__(control_step_count, costs, evolution_time,
                         hamiltonian, interpolation_policy,
                         operation_policy, system_step_multiplier)
        self.initial_states = initial_states
        self.magnus_policy = magnus_policy


class EvolveSchroedingerResult(object):
    """
    This class encapsulates the result of evolution
    under the schroedinger equation.
    
    Fields:
    final_states
    total_error
    """

    def __init__(self, final_states=None,
                 total_error=None):
        """
        See the class definition for arguments not listed here.
        """
        self.final_states = final_states
        self.toatl_error = None


class GrapeSchroedingerDiscreteState(GrapeState):
    """
    Fields:
    complex_controls
    control_count
    controls_shape
    costs
    dt
    final_control_step
    final_system_step
    hamiltonian
    hilbert_size
    initial_controls
    initial_states
    interpolation_policy
    iteration_count
    log_iteration_step
    magnus_policy
    max_control_norms
    operation_policy
    optimizer
    performance_policy
    save_file_path
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    system_step_multiplier
    """

    def __init__(self, complex_controls, control_count, control_step_count,
                 costs, evolution_time, hamiltonian, initial_controls,
                 initial_states, interpolation_policy, iteration_count, log_iteration_step,
                 magnus_policy,
                 max_control_norms, operation_policy, optimizer,
                 performance_policy,
                 save_file_path, save_iteration_step, system_step_multiplier,):
        """
        See class definition for arguments not listed here.

        Args:
        control_count
        control_step_count
        evolution_time
        """
        super().__init__(complex_controls, control_count, control_step_count,
                         costs, evolution_time, hamiltonian, initial_controls,
                         interpolation_policy, iteration_count,
                         log_iteration_step, max_control_norms, operation_policy,
                         optimizer, save_file_path, save_iteration_step,
                         system_step_multiplier,)
        self.hilbert_size = initial_states[0].shape[0]
        self.initial_states = initial_states
        self.magnus_policy = magnus_policy
        self.performance_policy = performance_policy


    def log_and_save(self, controls, error, grads, iteration, states):
        """
        If necessary, log to stdout and save to the save file.

        Args:
        controls :: ndarray - the optimization parameters
        error :: ndarray - the total error at the last time step
            of evolution
        grads :: ndarray - the current gradients of the cost function
            with resepct to controls
        iteration :: int - the optimization iteration

        states :: ndarray - the states at the last time step
            of evolution

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
            with h5py.File(self.save_file_path, "a") as save_file:
                save_file["controls"][save_step,] = controls
                save_file["error"][save_step,] = error
                save_file["grads"][save_step,] = grads
                save_file["states"][save_step,] = states


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
                save_file["control_step_count"] = self.control_step_count
                save_file["controls"] = np.zeros((save_count, self.control_step_count,
                                                  self.control_count,),
                                                 dtype=self.initial_controls.dtype)
                save_file["cost_names"] = np.array([np.string_("{}".format(cost))
                                                    for cost in self.costs])
                save_file["error"] = np.zeros((save_count),
                                              dtype=np.float64)
                save_file["evolution_time"]= self.evolution_time
                save_file["grads"] = np.zeros((save_count, self.control_step_count,
                                               self.param_count), dtype=self.initial_controls.dtype)
                save_file["initial_controls"] = self.initial_controls
                save_file["initial_states"] = self.initial_states
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                save_file["iteration_count"] = self.iteration_count
                save_file["magnus_policy"] = "{}".format(self.magnus_policy)
                save_file["max_control_norms"] = self.max_control_norms
                save_file["operation_policy"] = "{}".format(self.operation_policy)
                save_file["optimizer"] = "{}".format(self.optimizer)
                save_file["performance_policy"] = "{}".format(self.performance_policy)
                save_file["states"] = np.zeros((save_count, state_count,
                                                self.hilbert_size, 1),
                                               dtype=np.complex128)
                save_file["system_step_multiplier"] = self.system_step_multiplier
            #ENDWITH
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")


class GrapeSchroedingerResult():
    """
    This class encapsulates useful information about a
    grape optimization under the Schroedinger equation.

    Fields:
    """
    def __init__(self):
        """
        See class definition for arguments not listed here.

        Args:
        """
        pass
