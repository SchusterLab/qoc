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
    This class encapsulates data fields that are used by the
    qoc.core.lindbladdiscrete.evolve_lindblad_discrete program.

    Fields:
    control_eval_count
    control_eval_times
    cost_eval_step
    costs
    dt
    evolution_time
    final_system_eval_step
    hamiltonian
    initial_densities
    interpolation_policy
    lindblad_data
    method
    save_file_path
    save_intermediate_densities_
    step_cost_indices
    step_costs
    system_eval_count    
    """
    method = "evolve_lindblad_discrete"
    
    def __init__(self, control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian, initial_densities,
                 interpolation_policy,
                 lindblad_data, save_file_path, save_intermediate_densities_,
                 system_eval_count):
        """
        See class fields for arguments not listed here.
        """
        super().__init__(control_eval_count, cost_eval_step, costs,
                         evolution_time, hamiltonian, interpolation_policy,
                         save_file_path, system_eval_count)
        self.initial_densities = initial_densities
        self.lindblad_data = lindblad_data
        self.save_intermediate_densities_ = (save_intermediate_densities_
                                             and save_file_path is not None)

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
                save_file["initial_densities"] = self.initial_densities
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                if self.save_intermediate_densities:
                    save_file["intermediate_densities"] = np.zeros((self.system_eval_count,
                                                                    *self.initial_densities.shape),
                                                                   dtype=np.complex128)
                save_file["method"] = self.method
                save_file["system_eval_count"] = self.system_eval_count

    
    def save_intermediate_densities(self, densities, system_eval_step):
        """
        Save intermediate densities to the save file.
        """
        with h5py.File(self.save_file_path, "a") as save_file:
            save_file["intermediate_densities"][system_eval_step] = densities


class EvolveLindbladResult(object):
    """
    This class encapsulates the result of the
    qoc.core.lindbladdiscrete.evolve_lindblad_discrete program.
    
    Fileds:
    error
    final_densities
    """
    
    def __init__(self, error=None,
                 final_densities=None,):
        """
        See the class fields for arguments not listed here.
        """
        super().__init__()
        self.error = error
        self.final_densities = final_densities


class GrapeLindbladDiscreteState(GrapeState):
    """
    This class encapsulates the data fields that are used by the
    qoc.core.lindbladdiscrete.grape_lindblad_discrete program.

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
    impose_control_conditions
    initial_controls
    initial_densities
    interpolation_policy
    iteration_count
    lindblad_data
    log_iteration_step
    max_control_norms
    method
    min_error
    optimizer
    save_file_path
    save_intermediate_densities_
    save_iteration_step
    should_log
    should_save
    step_cost_indices
    step_costs
    system_eval_count
    """
    method = "grape_lindblad_discrete"
    save_intermediate_densities_ = False

    def __init__(self,
                 complex_controls,
                 control_count,
                 control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian,
                 impose_control_conditions,
                 initial_controls,
                 initial_densities,
                 interpolation_policy, iteration_count,
                 lindblad_data,
                 log_iteration_step, max_control_norms,
                 min_error, optimizer,
                 save_file_path, save_iteration_step,
                 system_eval_count,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__(complex_controls,
                 control_count,
                 control_eval_count, cost_eval_step, costs,
                 evolution_time, hamiltonian,
                 impose_control_conditions,
                 initial_controls,
                 interpolation_policy, iteration_count,
                 log_iteration_step, max_control_norms,
                 min_error, optimizer,
                 save_file_path, save_iteration_step,
                 system_eval_count,)
        self.hilbert_size = initial_densities[0].shape[0]
        self.initial_densities = initial_densities
        self.lindblad_data = lindblad_data
    

    def log_and_save(self, controls, error, final_densities, grads, iteration):
        """
        If necessary, log to stdout and save to the save file.

        Arguments:
        controls :: ndarray - the optimization parameters
        error :: ndarray - the total error at the last time step
            of evolution
        final_densities :: ndarray - the density matrices at the last time step
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
                save_file["error"][save_step,] = error
                save_file["final_densities"][save_step,] = final_densities
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
                save_file["control_eval_count"] = self.control_eval_count
                save_file["controls"] = np.zeros((save_count, self.control_eval_count,
                                                  self.control_count,),
                                                 dtype=self.initial_controls.dtype)
                save_file["cost_eval_step"] = self.cost_eval_step
                save_file["cost_names"] = np.array([np.string_("{}".format(cost))
                                                    for cost in self.costs])
                save_file["error"] = np.repeat(np.finfo(np.float64).max, save_count)
                save_file["evolution_time"]= self.evolution_time
                save_file["final_densities"] = np.zeros((save_count, density_count,
                                                         self.hilbert_size, self.hilbert_size),
                                                        dtype=np.complex128)
                save_file["grads"] = np.zeros((save_count, self.control_eval_count,
                                               self.control_count), dtype=self.initial_controls.dtype)
                save_file["initial_controls"] = self.initial_controls
                save_file["initial_densities"] = self.initial_densities
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                save_file["iteration_count"] = self.iteration_count
                save_file["max_control_norms"] = self.max_control_norms
                save_file["method"] = self.method
                save_file["optimizer"] = "{}".format(self.optimizer)
                save_file["system_eval_count"] = self.system_eval_count
            #ENDWITH
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")


class GrapeLindbladResult(object):
    """
    This class encapsulates the result of the
    qoc.core.lindbladdiscrete.grape_lindblad_discrete
    program.

    Fields:
    best_controls
    best_error
    best_final_densities
    best_iteration
    """
    def __init__(self, best_controls=None,
                 best_error=np.finfo(np.float64).max,
                 best_final_densities=None,
                 best_iteration=None,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__()
        self.best_controls = best_controls
        self.best_error = best_error
        self.best_final_densities = best_final_densities
        self.best_iteration = best_iteration
