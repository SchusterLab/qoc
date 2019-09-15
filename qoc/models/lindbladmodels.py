"""
lindbladmodels.py - a module to define classes to
encapsulate the necessary information to execute 
programs involving lindblad evolution
"""

import h5py
import numpy as np

from qoc.models.programstate import (GrapeState, ProgramState,)

class EvolveLindbladDiscreteState(ProgramState):
    """
    This class encapsulates the necessary information to evolve
    a set of density matrices under the lindblad equation and compute
    optimization error for one round.

    Fields:
    
    """
    
    def __init__(self, control_step_count,
                 costs, evolution_time,
                 hamiltonian, initial_densities,
                 interpolation_policy,
                 lindblad_data,
                 operation_policy,
                 system_step_multiplier,):

        """
        See class definition for arguments not listed here.

        Args:
        control_step_count :: int - the number of time steps at which the
            evolution time should be initially split into and the number
            of control parameter updates

        evolution_time :: float - the time over which the system will evolve
        """
        super().__init__(control_step_count, costs,
                         evolution_time, hamiltonian,
                         interpolation_policy,
                         operation_policy, system_step_multiplier)
        self.initial_densities = initial_densities
        self.lindblad_data = lindblad_data


class EvolveLindbladResult(object):
    """
    This class encapsulates the evolution of the
    Lindblad equation.
    
    Fileds:
    final_densities :: ndarray - the density matrices
        at the end of the evolution time
    total_error :: float - the optimization error
        incurred by the relevant cost functions
    """
    
    def __init__(self, final_densities=None,
                 total_error=None):
        """
        See the class definition for arguments not listed here.
        """
        self.final_densities = final_densities
        self.total_error = total_error


class GrapeLindbladDiscreteState(GrapeState):
    """
    This class encapsulates the necessary information
    to execute a grape progam.

    Fields:
    complex_controls
    control_count
    controls_shape
    costs
    dt
    final_control_step
    final_iteration
    final_system_step
    hamiltonian
    hilbert_size
    initial_controls
    initial_densities
    interpolation_policy
    iteration_count
    lindblad_data
    log_iteration_step
    max_control_norms
    minimum_error
    operation_policy
    optimizer
    save_file_path
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    system_step_multiplier
    """

    def __init__(self, complex_controls,
                 control_count,
                 control_step_count, costs, evolution_time,
                 hamiltonian, initial_controls,
                 initial_densities,
                 interpolation_policy, iteration_count,
                 lindblad_data,
                 log_iteration_step, max_control_norms,
                 minimum_error,
                 operation_policy, optimizer,
                 save_file_path, save_iteration_step,
                 system_step_multiplier,):
        """
        See class definition for arguments not listed here.
        """
        super().__init__(complex_controls,
                 control_count,
                 control_step_count, costs, evolution_time,
                 hamiltonian, initial_controls,
                 interpolation_policy, iteration_count,
                 log_iteration_step, max_control_norms,
                 minimum_error,
                 operation_policy, optimizer,
                 save_file_path, save_iteration_step,
                 system_step_multiplier,)
        self.hilbert_size = initial_densities[0].shape[0]
        self.initial_densities = initial_densities
        self.lindblad_data = lindblad_data
    

    def log_and_save(self, controls, densities, error, grads, iteration):
        """
        If necessary, log to stdout and save to the save file.

        Args:
        controls :: ndarray - the optimization parameters
        densities :: ndarray - the density matrices at the last time step
            of evolution
        error :: ndarray - the total error at the last time step
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
            with h5py.File(self.save_file_path, "a") as save_file:
                save_file["controls"][save_step,] = controls
                save_file["densities"][save_step,] = densities
                save_file["error"][save_step,] = error
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
            density_count = len(self.initial_densities)
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
                save_file["densities"] = np.zeros((save_count, density_count,
                                                   self.hilbert_size, self.hilbert_size),
                                                  dtype=np.complex128)
                save_file["error"] = np.zeros((save_count),
                                              dtype=np.float64)
                save_file["evolution_time"]= self.evolution_time
                save_file["grads"] = np.zeros((save_count, self.control_step_count,
                                               self.control_count), dtype=self.initial_controls.dtype)
                save_file["initial_controls"] = self.initial_controls
                save_file["initial_densities"] = self.initial_densities
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                save_file["iteration_count"] = self.iteration_count
                save_file["max_control_norms"] = self.max_control_norms
                save_file["operation_policy"] = "{}".format(self.operation_policy)
                save_file["optimizer"] = "{}".format(self.optimizer)
                save_file["system_step_multiplier"] = self.system_step_multiplier
            #ENDWITH
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")


class GrapeLindbladResult(object):
    """
    This class encapsulates useful information about a
    grape optimization under the Lindblad equation.

    Fields:
    best_controls
    best_final_densities
    best_iteration
    best_total_error
    """
    def __init__(self, best_controls=None,
                 best_final_densities=None,
                 best_iteration=None,
                 best_total_error=np.finfo(np.float64).max):
        """
        See class definition for arguments not listed here.
        """
        self.best_controls = best_controls
        self.best_final_densities = best_final_densities
        self.best_iteration = best_iteration
        self.best_total_error = best_total_error
