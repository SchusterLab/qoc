"""
programstate.py - a module to define classes that
encapsulate necessary information to execute the program
"""

import os

import autograd.numpy as anp
import h5py
import numpy as np

from qoc.core.maths import (magnus_m2_linear, magnus_m4_linear,
                            magnus_m6_linear,
                            magnus_m2_linear_param_indices, magnus_m4_linear_param_indices,
                            magnus_m6_linear_param_indices)
from .grapepolicy import GrapeSchroedingerPolicy
from .interpolationpolicy import InterpolationPolicy
from .magnuspolicy import MagnusPolicy
from .operationpolicy import OperationPolicy
from qoc.standard import ans_jacobian

### MAIN STRUCTURES ###

class GrapeState(object):
    """
    a class to encapsulate information to perform
    a GRAPE optimization.

    Fields:
    complex_controls :: bool - whether or not the parameters are complex
    control_count :: int - the number of parameters that should be supplied
        to the hamiltonian at each time step
    control_step_count :: int
    controls_shape :: int - the shape of the initial parameters
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    dt :: float - the length of a time step 
    evolution_time :: float - how long the system will evolve for
    final_iteration :: int
    final_control_step :: int 
    final_system_step :: int
    hilbert_size :: int - the dimension of the hilbert space in which
        states are evolving
    initial_controls :: ndarray - the parameters for the first
        optimization iteration
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: the number of iterations at which to print
        progress to stdout
    max_control_norms :: the maximum aboslute value at which to clip
        the parameters
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.Optimizer - an instance of an optimizer to perform
        gradient-based-optimization

    save_file_path :: str - the full path to the save file
    save_iteration_step :: the number of iterations at which to write
        progress to the save file
    should_log :: bool - whether or not to log progress
    should_save :: bool - whether or not to save progress
    step_costs :: [qoc.models.Cost] - the cost functions to guide optimization
        that need to be evaluated at each step
    step_cost_indices :: [int] - the indices into the costs list of the
        costs that need to be evaluated at every step
    system_step_multiplier :: int - the multiple of pulse_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps
    """
    def __init__(self, control_count,
                 control_step_count,
                 costs, evolution_time,
                 hilbert_size,
                 initial_controls,
                 iteration_count,
                 log_iteration_step,
                 max_control_norms, operation_policy,
                 optimizer, 
                 save_file_name,
                 save_iteration_step, save_path):
        """
        See class definition for argument specifications not listed here.

        Args:
        save_path :: str - the directory to create the save file in,
            the directory will be created if it does not exist
        save_file_name :: str - this will identify the save file
        """
        super().__init__()
        self.complex_controls = initial_controls.dtype in (np.complex64, np.complex128)
        self.control_count = control_count
        self.control_step_count = control_step_count
        self.controls_shape = (pulse_step_count, control_count)
        self.costs = costs
        self.dt = evolution_time / system_step_count
        self.evolution_time = evolution_time
        self.final_iteration = iteration_count - 1
        self.final_control_step = control_step_count - 1
        system_step_count = control_step_count * system_step_multiplier
        self.final_system_step = system_step_count - 1
        self.hilbert_size = hilbert_size
        self.initial_controls = initial_controls
        self.interpolation_policy = interpolation_policy
        self.iteration_count = iteration_count
        self.log_iteration_step = log_iteration_step
        self.max_control_norms = max_control_norms
        self.operation_policy = operation_policy
        self.optimizer = optimizer
        if save_iteration_step != 0 and save_path and save_file_name:
            self.save_file_path = _create_save_file_path(save_file_name, save_path)
            self.should_save = True
        else:
            self.save_file_path = None
            self.should_save = False
        self.save_iteration_step = save_iteration_step
        self.should_log = log_iteration_step != 0
        self.step_costs = list()
        self.step_cost_indices = list()
        for i, cost in enumerate(costs):
            if cost.requires_step_evaluation:
                self.step_costs.append(cost)
                self.step_cost_indices.append(i)
        #ENDFOR
        self.system_step_multiplier = system_step_multiplier


class GrapeResult(object):
    """
    This class encapsulates useful information about a GRAPE optimization.
    Fields:
    best_error :: ndarray - the total optimization error at the final time step
        of the iteration that achieved the lowest error
    best_grads :: ndarray - the gradients of the cost function with respect
        to the controls at the iteration that achieved the lowest error
    best_controls :: ndarray - the parameters at the iteration that achieved
        the lowest error
    best_states :: ndarray - the states at the final time step of the iteration
        that achieved the lowest error
    iteration :: int - the current iteration
    last_error :: ndarray - the total optimization error at the final time step
        of the last iteration
    last_grads :: ndarray - the gradients of the cost function with respect
        to the controls at the last iteration
    last_controls :: ndarray - the parameters at the last iteration
    last_states :: ndarray - the states at the final time step of the last iteration
    """
    def __init__(self, best_error=np.finfo(float).max, best_controls=None,
                 best_grads=None, best_states=None, iteration=0,
                 last_error=None,
                 last_grads=None, last_controls=None, last_states=None):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.best_error = best_error
        self.best_controls = best_controls
        self.best_grads = best_grads
        self.best_states = best_states
        self.iteration = 0
        self.last_error = last_error
        self.last_grads = last_grads
        self.last_controls = last_controls
        self.last_states = last_states


    def __str__(self):
        return ("best_error:{},\nbest_grads:\n{}\nbest_controls:\n{}\nbest_states:\n{}\n"
                "last_error:{},\nlast_grads:\n{}\nlast_controls:\n{}\nlast_states:\n{}"
                "".format(self.best_error, self.best_grads, self.best_controls, self.best_states,
                          self.last_error, self.last_grads, self.last_controls, self.last_states))


## Schroedinger Program States ##

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


## Lindblad Program States ##

class GrapeLindbladDiscreteState(GrapeState):
    """
    This class encapsulates the necessary information to optimize a set
    of time-discrete control parameters under the lindblad equation.

    Fields:
    complex_controls :: bool - whether or not the parameters are complex
    control_count :: int - the number of control parameters that should be supplied
        to the hamiltonian at each time step
    control_step_count :: int - the number of time steps at which the pulse
        should be optimized
    controls_shape :: int - the shape of the initial parameters
    costs :: iterable(qoc.models.Cost) - the cost functions to guide optimization
    dt :: float - the length of a time step 
    evolution_time :: float - the duration of the system evolution
    evolve_lindblad :: (controls :: ndarray, control_step :: int,
                        densities :: ndarray, dt :: float, time :: float
                        sentinel :: bool) -> densities :: ndarray
        - evolve the `densities` for the specified time step
    final_iteration :: int - the last optimization iteration
    final_control_step :: int 
    final_system_step :: int
    hilbert_size :: int - the dimension of the hilbert space in which
        density matrices are evolving
    initial_controls :: ndarray - the parameters for the first
        optimization iteration
    initial_densities :: ndarray - the states at the first time step
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: the number of iterations at which to print
        progress to stdout
    max_control_norms :: the maximum aboslute value at which to clip
        the parameters
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.Optimizer - an instance of an optimizer to perform
        gradient-based-optimization
    save_file_path :: str - the full path to the save file
    save_iteration_step :: the number of iterations at which to write
        progress to the save file
    should_log :: bool - whether or not to log progress
    should_save :: bool - whether or not to save progress
    step_costs :: [qoc.models.Cost] - the cost functions to guide optimization
        that need to be evaluated at each step
    step_cost_indices :: [int] - the indices into the costs list of the
        costs that need to be evaluated at every step
    system_step_multiplier :: int - the multiple of pulse_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps
    """
    def __init__(self, control_count,
                 control_step_count,
                 costs, evolution_time,
                 hamiltonian,
                 hilbert_size,
                 initial_controls,
                 initial_densities,
                 interpolation_policy,
                 iteration_count,
                 lindblad_operators,
                 log_iteration_step,
                 max_control_norms, operation_policy,
                 optimizer,
                 save_file_name,
                 save_iteration_step, save_path,
                 system_step_multiplier):
        """
        See class definiton for arguments not listed here.
        
        Args:
        hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
            - an autograd compatible function to generate the hamiltonian
              for the given controls and time
        lindblad_operators :: (time :: float) -> operators :: ndarray
            - a function to generate the lindblad operators for a given time,
              an array of operators should be returned even if there is only
              one operator
        save_file_name :: str - this will identify the save file
        save_path :: str - the directory to create the save file in,
            the directory will be created if it does not exist
        """
        super().__init__(control_count,
                         control_step_count,
                         costs, evolution_time,
                         hilbert_size,
                         initial_controls,
                         iteration_count,
                         log_iteration_step,
                         max_control_norms, operation_policy,
                         optimizer,
                         save_file_name,
                         save_iteration_step, save_path,
                         system_step_multiplier)
        self.evolve_lindblad = _choose_evolve_lindblad(hamiltonian,
                                                       interpolation_policy,
                                                       lindblad_operators)
        self.initial_densities = initial_densities


class EvolveLindbladDiscreteState(object):
    """
    This class encapsulates the necessary information to evolve
    a set of density matrices under the lindblad equation and compute
    optimization error for one round.

    Fields:
    controls :: ndarray - the controls that should be provided to the
        hamiltonian for the evolution
    costs :: iterable(qoc.models.Cost) - the cost functions to guide optimization
    dt :: float - the length of a time step 
    evolve_lindblad :: (controls :: ndarray, control_step :: int,
                        densities :: ndarray, dt :: float, time :: float
                        sentinel :: bool) -> densities :: ndarray
        - evolve the `densities` for the specified time step
    final_control_step :: int 
    final_system_step :: int
    initial_densities :: ndarray - the states at the first time step
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    system_step_multiplier :: int - the multiple of pulse_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps    
    """

    def __init__(self, control_step_count,
                 controls, costs, evolution_time,
                 hamiltonian, initial_densities,
                 interpolation_policy,
                 lindblad_operators,
                 operation_policy,
                 system_step_multiplier):
        """
        See class definition for arguments not listed here.

        Args:
        hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
            - an autograd compatible function to generate the hamiltonian
              for the given controls and time
        interpolation_policy :: qoc.InterpolationPolicy - how parameters
            should be interpolated for intermediate time steps
        lindblad_operators :: (time :: float) -> operators :: ndarray
            - a function to generate the lindblad operators for a given time,
              an array of operators should be returned even if there is only
              one operator
        """
        self.controls = controls
        self.costs = costs
        system_step_count = control_step_count * system_step_multiplier
        self.dt = evolution_time / system_step_count
        self.evolve_lindblad = _choose_evolve_lindblad(hamiltonian,
                                                       interpolation_policy,
                                                       lindblad_operators)
        self.final_control_step = control_step_count - 1
        self.final_system_step = system_step_count - 1
        self.initial_densities = initial_densities
        self.operation_policy = operation_policy
        self.system_step_multiplier = system_step_multiplier


class EvolveLindbladDiscreteResult(object):
    """
    This class encapsulates the evolution of the
    Lindblad equation under time-discerete controls.
    
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


### HELPER METHODS ###

def _choose_evolve_lindblad(hamiltonian, interpolation_policy,
                            lindblad_operators):
    """
    Generate a function which will evolve a set of density matrices
    under the lindbladian specified by the given hamiltonian and
    lindblad operators.

    Args:
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_operators :: (time :: float) -> operators :: ndarray
        - a function to generate the lindblad operators for a given time,
          an array of operators should be returned even if there is only
          one operator

    Returns:
    evolve_lindblad :: (controls :: ndarray, control_step :: int,
                        densities :: ndarray, dt :: float, time :: float
                        sentinel :: bool) -> densities :: ndarray
        - evolve the `densities` for the specified time step
    """
    pass


def _choose_magnus(hamiltonian, interpolation_policy, magnus_policy):
    """
    Choose a magnus expansion method based on a magnus policy and corresponding
    interpolation policy. Also, create the gradient function for the
    magnus expansion with respect to the parameters.
    The controls argument could be large and we want to avoid propagating unnecessary
    gradients. So, the dmagnus_dcontrols function will return the magnus expansion,
    the gradient of the magnus expansion with respect to the parameters, and
    the indices of the parameters with nonzero gradients.

    Args:
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - the time and parameter dependent hamiltonian

    Returns:
    dmagnus_dcontrols :: (dt :: float, controls :: np.ndarray, step :: int, time :: float)
                       -> (dmagnus_dcontrols :: ndarray, indices :: ndarray,
                           magnus :: ndarray)
        - the gradient of the magnus expansion with respect to the parameters--including only
          nonzero gradients, the indices of parameters in the controls array with nonzero gradients,
          and the magnus expansion
    magnus :: (dt :: float, controls :: np.ndarray, step :: int, time :: float)
               -> magnus :: ndarray
        - the magnus expansion
    magnus_param_indices :: (dt :: float, controls :: np.ndarray, step :: int, time :: float)
                             -> magnus_param_indices :: ndarray
        - This function returns the indices of controls that should be passed to magnus
          and dmagnus_dcontrols
    """
    if interpolation_policy == InterpolationPolicy.LINEAR:
        if magnus_policy == MagnusPolicy.M2:
            magnus = (lambda *args, **kwargs:
                      magnus_m2_linear(hamiltonian, *args, **kwargs))
            magnus_param_indices = (lambda *args, **kwargs:
                                    magnus_m2_linear_param_indices(hamiltonian, *args, **kwargs))
        elif magnus_policy == MagnusPolicy.M4:
            magnus = (lambda *args, **kwargs:
                      magnus_m4_linear(hamiltonian, *args, **kwargs))
            magnus_param_indices = (lambda *args, **kwargs:
                                    magnus_m4_linear_param_indices(hamiltonian,
                                                                   *args, **kwargs))
        else:
            magnus = (lambda *args, **kwargs:
                      magnus_m6_linear(hamiltonian, *args, **kwargs))
            magnus_param_indices = (lambda *args, **kwargs:
                                    magnus_m6_linear_param_indices(hamiltonian,
                                                                   *args, **kwargs))
    #ENDIF
    
    magnus_wrapper = lambda *args, **kwargs: anp.real(magnus(*args, **kwargs))
    dmagnus_dcontrols = None

    return dmagnus_dcontrols, magnus, magnus_param_indices


def _create_save_file_path(save_file_name, save_path):
    """
    Create the full path to a h5 file using the base name
    save_file_name in the path save_path. File name conflicts are avoided
    by appending a numeric prefix to the file name. This method assumes
    that all objects in save_path that contain _{save_file_name}.h5
    are created with this convention.
    Args:
    save_file_name :: str - see GrapeState.__init__
    save_path :: str - see GrapeState.__init__

    Returns:
    save_file_path :: str - see GrapeState fields
    """
    # Ensure the path exists.
    os.makedirs(save_path, exist_ok=True)
    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory. 
    max_numeric_prefix = -1
    for file_name in os.listdir(save_path):
        if ("_{}.h5".format(save_file_name)) in file_name:
            max_numeric_prefix = max(int(file_name.split("_")[0]),
                                     max_numeric_prefix)
    #ENDFOR
    save_file_name_augmented = ("{:05d}_{}.h5"
                                "".format(max_numeric_prefix + 1,
                                          save_file_name))
    return os.path.join(save_path, save_file_name_augmented)

