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


class GrapeSchroedingerDiscreteState(GrapeState):
    """
    a class to encapsulate the necessary information to perform a
    schroedinger, discrete GRAPE optimization.
    Fields:
    complex_controls :: bool - whether or not the optimization parameters
        are complex
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    dt :: float - the length of a time step 
    dmagnus_dcontrols :: (dt :: float, controls :: ndarray, step :: int, time :: float)
                               -> (magnus :: ndarray,
                                   dmagnus_dcontrols :: ndarray)
        - This function evaluates the magnus expansion and the gradient of the magnus
          expansion with respect to the controls argument. The controls argument
          consists of all controls specified by magnus_param_indices.
    final_iteration :: int - the last optimization iteration
    final_time_step :: int - the last pulse step, i.e. point where
        parameters are updated
    grape_schroedinger_policy :: qoc.GrapeSchroedingerPolicy - specification
        for how to perform the main integration
    hilbert_size :: int - the dimension of the hilbert space that the evolving
        states live in
    initial_controls :: ndarray - the parameters for the first iteration
        of optimization
    initial_states :: ndarray - the states at the initial time step
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: the number of iterations at which to print
        progress to stdout
    magnus :: (dt :: float, controls :: ndarray, step :: int, time :: float)
               -> magnus :: ndarray
        - This function evaluates the magnus expansion. The controls argument
          consist of all controls specified by magnus_param_indices.
    magnus_control_indices :: (dt :: float, parms :: ndarray, step :: int, time :: float)
                            -> magnus_control_indices :: array
        - This function returns the param indices that should be included
        in the controls argument to "magnus". The point of this paradigm is
        to figure out which controls should be sent to the magnus expansion
        to be used for interpolation. That way, we only have to calculate
        the gradient of the magnus expansion with respect to he controls used.
        In this way, we still keep the abstraction that any number of controls
        may be used for interpolation. In practice, we expect that only a few
        of the controls near the index "step" will be used. Therefore, we expect
        to save memory and time.
    magnus_policy :: qoc.MagnusPolicy - specify how to perform the 
        magnus expansion
    max_control_norms :: the maximum aboslute value at which to clip
        the parameters
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.Optimizer - an instance of an optimizer to perform
        gradient-based-optimization
    control_count :: int - the number of control parameters required at each
         optimization time step
    controls_shape :: int - the shape of the initial parameters
    pulse_step_count :: int - the number of time steps at which the pulse
        should be optimized
    pulse_time :: float - the duration of the control pulse
    save_file_path :: str - the full path to the save file
    save_iteration_step :: the number of iterations at which to write
        progress to the save file
    should_log :: bool - whether or not to log progress
    should_save :: bool - whether or not to save progress
    step_cost_indices :: [int] - the indices into the costs list of the
        costs that need to be evaluated at every step
    system_step_multiplier :: int - the multiple of pulse_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps
    """
    
    def __init__(self, costs, grape_schroedinger_policy,
                 hamiltonian, hilbert_size,
                 initial_controls,
                 initial_states,
                 interpolation_policy,
                 iteration_count, log_iteration_step,
                 magnus_policy, max_param_norms, operation_policy,
                 optimizer, param_count, pulse_step_count, pulse_time,
                 save_file_name, save_iteration_step, save_path,
                 system_step_multiplier):
        """
        See class definition for argument specifications not listed here.
        Args:
        hamiltonian :: (controls :: ndarray, time :: float)
                        -> hamiltonian :: ndarray
            - an autograd compatible function that returns the system
              hamiltonian for a specified time and optimization parameters
        save_file_name :: str - this will identify the save file
        save_path :: str - the directory to create the save file in,
            the directory will be created if it does not exist
        """
        super().__init__(costs, hilbert_size, initial_controls,
                         iteration_count,
                         log_iteration_step, max_param_norms,
                         operation_policy, optimizer, param_count,
                         pulse_step_count, pulse_time, 
                         save_file_name, save_iteration_step,
                         save_path)

        (self.dmagnus_dcontrols,
         self.magnus,
         self.magnus_param_indices) = _choose_magnus(hamiltonian,
                                                     interpolation_policy,
                                                     magnus_policy)
        self.final_iteration = iteration_count - 1
        self.final_pulse_step = pulse_step_count - 1
        system_step_count = pulse_step_count * system_step_multiplier
        self.final_time_step = system_step_count - 1
        self.dt = pulse_time / system_step_count
        self.grape_schroedinger_policy = grape_schroedinger_policy
        self.initial_states = initial_states
        self.interpolation_policy = interpolation_policy
        self.magnus_policy = magnus_policy
        self.system_step_multiplier = system_step_multiplier


    def log_and_save(self, error, grads, iteration, controls, states):
        """
        If necessary, log to stdout and save to the save file.
        Args:
        error :: ndarray - the total error at the last time step
            of evolution
        grads :: ndarray - the current gradients of the cost function
            with resepct to controls
        iteration :: int - the optimization iteration
        controls :: ndarray - the optimization parameters
        states :: ndarray - the states at the last time step
            of evolution
        Returns: none
        """
        # Don't log if the iteration number is invalid.
        if iteration > self.final_iteration:
            return

        # Determine if it is the final iteration.
        is_final_iteration = iteration == self.final_iteration
        
        if (self.should_log
            and (np.mod(iteration, self.log_iteration_step) == 0
                 or is_final_iteration)):
            grads_norm = np.linalg.norm(grads)
            print("{:^6d} | {:^1.8e} | {:^1.8e}"
                  "".format(iteration, error,
                            grads_norm))

        # The one-liner here is jank but it saves doing another
        # integer division. If the iteration is a multiple of save_iteration_step,
        # the long one-liner condition will succeed and save_step will be
        # the nth save step.
        if (self.should_save
            and (np.mod(iteration, self.save_iteration_step) == 0
                 or is_final_iteration)):
            save_step, _ = np.divmod(iteration, self.save_iteration_step)
            with h5py.File(self.save_file_path, "a") as save_file:
                save_file["error"][save_step,] = error
                save_file["grads"][save_step,] = grads
                save_file["controls"][save_step,] = controls
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
                save_file["cost_names"] = np.array([np.string_("{}".format(cost))
                                                    for cost in self.costs])
                save_file["error"] = np.zeros((save_count),
                                              dtype=np.float64)
                save_file["grads"] = np.zeros((save_count, self.pulse_step_count,
                                               self.param_count), dtype=self.initial_controls.dtype)
                save_file["grape_schroedinger_policy"] = "{}".format(self.grape_schroedinger_policy)
                save_file["initial_controls"] = self.initial_controls
                save_file["initial_states"] = self.initial_states
                save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
                save_file["magnus_policy"] = "{}".format(self.magnus_policy)
                save_file["max_param_norms"] = self.max_param_norms
                save_file["operation_policy"] = "{}".format(self.operation_policy)
                save_file["optimizer"] = "{}".format(self.optimizer)
                save_file["controls"] = np.zeros((save_count, self.pulse_step_count,
                                                self.param_count,), dtype=self.initial_controls.dtype)
                save_file["param_count"] = self.param_count
                save_file["pulse_step_count"] = self.pulse_step_count
                save_file["pulse_time"]= self.pulse_time
                save_file["states"] = np.zeros((save_count, state_count,
                                                self.hilbert_size, 1),
                                               dtype=np.complex128)
                save_file["system_step_multiplier"] = self.system_step_multiplier
            #ENDWITH
        #ENDIF

        if self.should_log:
            print("iter   |   total error  |    grads_l2   \n"
                  "=========================================")


### HELPER METHODS ###

# def _create_save_file_path(save_file_name, save_path):
#     """
#     Create the full path to a h5 file using the base name
#     save_file_name in the path save_path. File name conflicts are avoided
#     by appending a numeric prefix to the file name. This method assumes
#     that all objects in save_path that contain _{save_file_name}.h5
#     are created with this convention.
#     Args:
#     save_file_name :: str - see GrapeState.__init__
#     save_path :: str - see GrapeState.__init__

#     Returns:
#     save_file_path :: str - see GrapeState fields
#     """
#     # Ensure the path exists.
#     os.makedirs(save_path, exist_ok=True)
#     # Create a save file name based on the one given; ensure it will
#     # not conflict with others in the directory. 
#     max_numeric_prefix = -1
#     for file_name in os.listdir(save_path):
#         if ("_{}.h5".format(save_file_name)) in file_name:
#             max_numeric_prefix = max(int(file_name.split("_")[0]),
#                                      max_numeric_prefix)
#     #ENDFOR
#     save_file_name_augmented = ("{:05d}_{}.h5"
#                                 "".format(max_numeric_prefix + 1,
#                                           save_file_name))
#     return os.path.join(save_path, save_file_name_augmented)
