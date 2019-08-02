"""
grapestate.py - a module for classes to encapsulate the state of a
GRAPE optimization
"""

import os

import autograd.numpy as anp
import h5py
import numpy as np

from qoc.core.maths import (magnus_m2_linear, magnus_m4_linear,
                            magnus_m6_linear,
                            magnus_m2_linear_param_indices, magnus_m4_linear_param_indices,
                            magnus_m6_linear_param_indices)
from qoc.models.grapepolicy import GrapeSchroedingerPolicy
from qoc.models.interpolationpolicy import InterpolationPolicy
from qoc.models.magnuspolicy import MagnusPolicy
from qoc.models.operationpolicy import OperationPolicy
from qoc.util import ans_jacobian

### MAIN STRUCTURES ###

class GrapeState(object):
    """
    a class to encapsulate information to perform
    a GRAPE optimization.
    Fields:
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    cost_count :: int - the number of cost functions
    hilbert_size :: int - the dimension of the hilbert space in which
        states are evolving
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: the number of iterations at which to print
        progress to stdout
    max_param_amplitudes :: the maximum aboslute value at which to clip
        the parameters
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.Optimizer - an instance of an optimizer to perform
        gradient-based-optimization
    pulse_step_count :: int - the number of time steps at which the pulse
        should be optimized
    pulse_time :: float - the duration of the control pulse
    save_file_path :: str - the full path to the save file
    save_iteration_step :: the number of iterations at which to write
        progress to the save file
    should_log :: bool - whether or not to log progress
    should_save :: bool - whether or not to save progress
    step_costs :: [qoc.models.Cost] - the cost functions to guide optimization
        that need to be evaluated at each step
    step_cost_indices :: [int] - the indices into the costs list of the
        costs that need to be evaluated at every step
    """

    def __init__(self, costs, hilbert_size, iteration_count,
                 log_iteration_step,
                 max_param_amplitudes, operation_policy,
                 optimizer, pulse_step_count,
                 pulse_time, save_file_name,
                 save_iteration_step, save_path):
        """
        See class definition for argument specifications not listed here.
        Args:
        save_path :: str - the directory to create the save file in,
            the directory will be created if it does not exist
        save_file_name :: str - this will identify the save file
        """
        super().__init__()
        self.costs = costs
        self.cost_count = len(costs)
        self.hilbert_size = hilbert_size
        self.iteration_count = iteration_count
        self.log_iteration_step = log_iteration_step
        self.max_param_amplitudes = max_param_amplitudes
        self.pulse_step_count = pulse_step_count
        self.pulse_time = pulse_time
        self.operation_policy = operation_policy
        self.optimizer = optimizer
        if save_iteration_step != 0 and save_path and save_file_name:
            self.save_file_path = _create_save_file_path(save_file_name, save_path)
        else:
            self.save_file_path = None
        self.save_iteration_step = save_iteration_step
        self.should_log = log_iteration_step != 0
        self.should_save = save_iteration_step != 0
        self.step_costs = list()
        self.step_cost_indices = list()
        for i, cost in enumerate(costs):
            if cost.requires_step_evaluation:
                self.step_costs.append(cost)
                self.step_cost_indices.append(i)
        #ENDFOR


class GrapeSchroedingerDiscreteState(GrapeState):
    """
    a class to encapsulate the necessary information to perform a
    schroedinger, discrete GRAPE optimization.
    Fields:
    costs :: [qoc.models.Cost] - the cost functions to guide optimization
    cost_count :: int - the number of cost functions
    dt :: float - the length of a time step 
    dmagnus_dparams :: (dt :: float, params :: numpy.ndarray, step :: int, time :: float)
                               -> (magnus :: numpy.ndarray,
                                   dmagnus_dparams :: numpy.ndarray)
        - This function evaluates the magnus expansion and the gradient of the magnus
          expansion with respect to the params argument. The params argument
          consists of all params specified by magnus_param_indices.
    final_time_step :: int - the last evolution time step
    grape_schroedinger_policy :: qoc.GrapeSchroedingerPolicy - specification
        for how to perform the main integration
    hilbert_size :: int - the dimension of the hilbert space that the evolving
        states live in
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    iteration_count :: int - the number of iterations to optimize for
    log_iteration_step :: the number of iterations at which to print
        progress to stdout
    magnus :: (dt :: float, params :: numpy.ndarray, step :: int, time :: float)
               -> magnus :: numpy.ndarray
        - This function evaluates the magnus expansion. The params argument
          consist of all params specified by magnus_param_indices.
    magnus_param_indices :: (dt :: float, parms :: numpy.ndarray, step :: int, time :: float)
                            -> magnus_param_indices :: numpy.array
        - This function returns the param indices that should be included
        in the params argument to "magnus". The point of this paradigm is
        to figure out which params should be sent to the magnus expansion
        to be used for interpolation. That way, we only have to calculate
        the gradient of the magnus expansion with respect to he params used.
        In this way, we still keep the abstraction that any number of params
        may be used for interpolation. In practice, we expect that only a few
        of the params near the index "step" will be used. Therefore, we expect
        to save memory and time.
    magnus_policy :: qoc.MagnusPolicy - specify how to perform the 
        magnus expansion
    max_param_amplitudes :: the maximum aboslute value at which to clip
        the parameters
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    optimizer :: qoc.Optimizer - an instance of an optimizer to perform
        gradient-based-optimization
    param_count :: int - the number of control parameters required at each
         optimization time step
    pulse_step_count :: int - the number of time steps at which the pulse
        should be optimized
    pulse_time :: float - the duration of the control pulse
    save_file_path :: str - the full path to the save file
    save_iteration_step :: the number of iterations at which to write
        progress to the save file
    step_cost_indices :: [int] - the indices into the costs list of the
        costs that need to be evaluated at every step
    system_step_multiplier :: int - the multiple of pulse_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps
    """
    
    def __init__(self, costs, grape_schroedinger_policy,
                 hamiltonian, hilbert_size, interpolation_policy,
                 iteration_count, log_iteration_step,
                 magnus_policy, max_param_amplitudes, operation_policy,
                 optimizer, param_count, pulse_step_count, pulse_time,
                 save_file_name, save_iteration_step, save_path,
                 system_step_multiplier):
        """
        See class definition for argument specifications not listed here.
        Args:
        hamiltonian :: (params :: numpy.ndarray, time :: float)
                        -> hamiltonian :: numpy.ndarray
            - an autograd compatible function that returns the system
              hamiltonian for a specified time and optimization parameters
        save_file_name :: str - this will identify the save file
        save_path :: str - the directory to create the save file in,
            the directory will be created if it does not exist
        """
        super().__init__(costs, hilbert_size, iteration_count,
                         log_iteration_step, max_param_amplitudes,
                         operation_policy, optimizer,
                         pulse_step_count, pulse_time, 
                         save_file_name, save_iteration_step,
                         save_path)

        (self.dmagnus_dparams,
         self.magnus,
         self.magnus_param_indices) = _choose_magnus(hamiltonian,
                                                     interpolation_policy,
                                                     magnus_policy)
        self.final_time_step = (pulse_step_count - 1) * system_step_multiplier
        self.dt = pulse_time / self.final_time_step
        self.grape_schroedinger_policy = grape_schroedinger_policy
        self.interpolation_policy = interpolation_policy
        self.magnus_policy = magnus_policy
        self.param_count = param_count
        self.system_step_multiplier = system_step_multiplier


    def log(self, error, grads, iteration, params, states):
        """
        Log to stdout.
        Args:
        error :: numpy.ndarray - the total error at the last time step
            of evolution
        grads :: numpy.ndarray - the current gradients of the cost function
            with resepct to params
        iteration :: int - the optimization iteration
        params :: numpy.ndarray - the optimization parameters
        states :: numpy.ndarray - the states at the last time step
            of evolution
        Returns: none
        """
        grads_norm = np.linalg.norm(grads)
        print("{:^6d} | {:^1.8e} | {:^1.8e}"
              "".format(iteration, error,
                        grads_norm))


    def log_initial(self):
        """
        Log the initial header.
        Args: none
        Returns: none
        """
        print("iter   |   total error  |    grads_l2   \n"
              "=========================================")


    def save(self, error, grads, iteration, params, states):
        """
        Save to the save file.
        Args:
        error :: numpy.ndarray - the total error at the last time step
            of evolution
        grads :: numpy.ndarray - the current gradients of the cost function
            with resepct to params
        iteration :: int - the optimization iteration
        params :: numpy.ndarray - the optimization parameters
        states :: numpy.ndarray - the states at the last time step
            of evolution
        Returns: none
        """
        save_step, _ = np.divmod(iteration,
                                 self.save_iteration_step)
        with h5py.File(self.save_file_path, "a") as save_file:
            save_file["error"][save_step,] = error
            save_file["grads"][save_step,] = grads
            save_file["params"][save_step,] = params
            save_file["states"][save_step,] = states


    def save_initial(self, initial_params, initial_states):
        """
        Save all initial values to the save file.
        Create necessary data sets.
        Args:
        initial_params :: numpy.ndarray - the initial parameters of optimization
        initial_states :: numpy.ndarray - the initial states to propagate
        Returns: none
        """
        # Notify the user where the file is being saved.
        print("QOC is saving this optimization run to {}."
              "".format(self.save_file_path))
        
        save_count, save_count_remainder = np.divmod(self.iteration_count, self.save_iteration_step)
        # If the final iteration doesn't fall on a save step, add a save step.
        if save_count_remainder != 0:
            save_count += 1
        state_count = len(initial_states)
        
        with h5py.File(self.save_file_path, "w") as save_file:
            save_file["cost_names"] = np.array([np.string_("{}".format(cost))
                                                for cost in self.costs])
            save_file["error"] = np.zeros((save_count, self.cost_count),
                                          dtype=np.float64)
            save_file["grads"] = np.zeros((save_count, self.pulse_step_count,
                                           self.param_count), dtype=initial_params.dtype)
            save_file["grape_schroedinger_policy"] = "{}".format(self.grape_schroedinger_policy)
            save_file["initial_params"] = initial_params
            save_file["initial_states"] = initial_states
            save_file["interpolation_policy"] = "{}".format(self.interpolation_policy)
            save_file["magnus_policy"] = "{}".format(self.magnus_policy)
            save_file["max_param_amplitudes"] = "{}".format(self.magnus_policy)
            save_file["operation_policy"] = "{}".format(self.operation_policy)
            save_file["optimizer"] = "{}".format(self.optimizer)
            save_file["params"] = np.zeros((save_count, self.pulse_step_count,
                                            self.param_count,), dtype=initial_params.dtype)
            save_file["param_count"] = self.param_count
            save_file["pulse_step_count"] = self.pulse_step_count
            save_file["pulse_time"]= self.pulse_time
            save_file["states"] = np.zeros((save_count, state_count,
                                            self.hilbert_size, 1),
                                           dtype=np.complex128)
            save_file["system_step_multiplier"] = self.system_step_multiplier
            save_file.close()
        #ENDIF
        

class GrapeResult(object):
    """
    This class encapsulates useful information about a GRAPE optimization.
    Fields:
    error :: numpy.ndarray - the total optimization error at the final time step
        of the last iteration
    grads :: numpy.ndarray - the gradients of the cost function with respect
        to the params at the last iteration
    params :: numpy.ndarray - the parameters at the last iteration
    states :: numpy.ndarray - the states at the final time step of the last iteration
    """
    def __init__(self, error, grads,
                 params, states):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.error = error
        self.grads = grads
        self.params = params
        self.states = states


class EvolveResult(object):
    """
    a class to encapsulate the results of an evolution
    Fields:
    final_states :: [numpy.ndarray] - the resultant, evolved final states
    """
    def __init__(self, final_states):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.final_states = final_states


### HELPER METHODS ###

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


def _choose_magnus(hamiltonian, interpolation_policy, magnus_policy):
    """
    Choose a magnus expansion method based on a magnus policy and corresponding
    interpolation policy. Also, create the gradient function for the
    magnus expansion with respect to the parameters.
    The params argument could be large and we want to avoid propagating unnecessary
    gradients. So, the dmagnus_dparams function will return the magnus expansion,
    the gradient of the magnus expansion with respect to the parameters, and
    the indices of the parameters with nonzero gradients.
    The shape of the jacobian of the magnus expansion can be obtained
    by rolling the axes of the autograd jacobian backwards by one.
    Autograd defines the shape of the jacobian in terms of the
    original function's input and the output shapes
    (https://github.com/HIPS/autograd/blob/master/autograd/differential_operators.py),
    but this is not what we want
    (https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).
    Args:
    hamiltonian :: (params :: numpy.ndarray, time :: float) -> hamiltonian :: numpy.ndarray
        - the time and parameter dependent hamiltonian
    Returns:
    dmagnus_dparams :: (dt :: float, params :: np.ndarray, step :: int, time :: float)
                       -> (dmagnus_dparams :: numpy.ndarray, indices :: numpy.ndarray,
                           magnus :: numpy.ndarray)
        - the gradient of the magnus expansion with respect to the parameters--including only
          nonzero gradients, the indices of parameters in the params array with nonzero gradients,
          and the magnus expansion
    magnus :: (dt :: float, params :: np.ndarray, step :: int, time :: float)
               -> magnus :: numpy.ndarray
        - the magnus expansion
    magnus_param_indices :: (dt :: float, params :: np.ndarray, step :: int, time :: float)
                             -> magnus_param_indices :: numpy.ndarray
        - This function returns the indices of params that should be passed to magnus
          and dmagnus_dparams
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
    def dmagnus_dparams(*args, **kwargs):
        _magnus, _dmagnus_dparams = ans_jacobian(magnus_wrapper, 1)(*args, **kwargs)
        old_axes = np.arange(len(_dmagnus_dparams.shape))
        new_axes = np.roll(old_axes, -2)
        return (np.moveaxis(_dmagnus_dparams, old_axes, new_axes),
                _magnus)

    return dmagnus_dparams, magnus, magnus_param_indices
