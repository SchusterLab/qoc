"""
grapestate.py - a module for classes to encapsulate the state of a
GRAPE optimization
"""

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
    step_costs :: [qoc.models.Cost] - the cost functions to guide optimization
        that need to be evaluated at each step
    step_cost_indices :: [int] - the indices into the costs list of the
        costs that need to be evaluated at every step
    """

    def __init__(self, costs, hilbert_size, iteration_count,
                 log_iteration_step,
                 max_param_amplitudes, operation_policy,
                 optimizer, pulse_step_count,
                 pulse_time, save_file_path,
                 save_iteration_step, step_costs):
        """
        See class definition for argument specifications not listed here.
        Args:
        save_path :: str - the directory to create the save file in,
            the directory will be created if it does not exist
        save_file_name :: str - this will identify the save file
        """
        super().__init__()
        self.costs = costs
        self.hilbert_size = hilbert_size
        self.iteration_count = iteration_count
        self.log_iteration_step = log_iteration_step
        self.max_param_amplitudes = max_param_amplitudes
        self.pulse_step_count = pulse_step_count
        self.pulse_time = pulse_time
        self.operation_policy = operation_policy
        self.optimizer = optimizer
        if save_iteration_step != 0 and save_path and save_file_name:
            self.save_file_path = _create_save_file_path(save_path, save_file_name)
        else:
            self.save_file_path = None
        self.save_iteration_step = save_iteration_step
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
    dmagnus_dparams :: (dt :: float, params :: numpy.ndarray, step :: int, time :: float)
                               -> (magnus :: numpy.ndarray,
                                   dmagnus_dparams :: numpy.ndarray)
        - This function evaluates the magnus expansion and the gradient of the magnus
          expansion with respect to the params argument. The params argument
          consists of all params specified by magnus_param_indices.
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
                         save_file_name, save_iteration_step, save_path,)
        (self.dmagnus_dparams,
         self.magnus,
         self.magnus_param_indices) = _choose_magnus(hamiltonian,
                                                     interpolation_policy,
                                                     magnus_policy)
        self.grape_schroedinger_policy = grape_schroedinger_policy
        self.interpolation_policy = interpolation_policy
        self.magnus_policy = magnus_policy
        self.param_count = param_count
        self.system_step_multiplier = system_step_multiplier


    def save_initial(self, initial_states, initial_params):
        """
        save all initial values to the save file
        Args:
        initial_states :: numpy.ndarray - the initial states to propagate
        initial_params :: numpy.ndarray - the initial parameters of optimization
        Returns: none
        """
        if self.save_iteration_step != 0:
            with h5py.File(self.save_file_path, "w") as save_file:
                save_file.create_dataset("cost_names",
                                         data=["{}".format(cost) for cost in self.costs])
                save_file.create_dataset("grape_schroedinger_policy",
                                         data="{}".format(self.grape_schroedinger_policy))
                save_file.create_dataset("initial_params", data=initial_params)
                save_file.create_dataset("initial_states", data=initial_states)
                save_file.create_dataset("interpolation_policy",
                                         data="{}".format(self.interpolation_policy))
                save_file.create_dataset("iteration_count", data=self.iteration_count)
                save_file.create_dataset("magnus_policy",
                                         data="{}".format(self.magnus_policy))
                save_file.create_dataset("max_param_amplitudes",
                                         data=self.max_param_amplitudes)
                save_file.create_dataset("operation_policy",
                                         data="{}".format(self.operation_policy))
                save_file.create_dataset("optimizer", data="{}".format(self.optimizer))
                save_file.create_dataset("param_count", data=self.param_count)
                save_file.create_dataset("pulse_step_count", data=self.pulse_step_count)
                save_file.create_dataset("pulse_time", data=self.pulse_time)
                save_file.create_dataset("system_step_multiplier",
                                         data=self.system_step_multiplier)
        #ENDIF
        

class GrapeResult(object):
    """
    a class to encapsulate the results of a GRAPE optimization
    Fields:
    states :: [numpy.ndarray] - the resultant, evolved final states
    params :: numpy.ndarray - the resultant, optimized params
    error :: numpy.ndarray - the final error of each cost function
    """
    def __init__(self, _states, params, error):
        """
        See class definition for argument specifications.
        """
        super().__init__()
        self.states = states
        self.params = params
        self.error = error

        
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
        if file_name.contains("_{}.h5".format(save_file_name)):
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
            magnus = lambda *args, **kwargs: magnus_m2_linear(hamiltonian, *args, **kwargs)
            magnus_param_indices = (lambda *args, **kwargs:
                                    magnus_m2_linear_param_indices(hamiltonian, *args, **kwargs))
        elif magnus_policy == MagnusPolicy.M4:
            magnus = lambda *args, **kwargs: magnus_m4_linear(hamiltonian, *args, **kwargs)
            magnus_param_indices = (lambda *args, **kwargs:
                                    magnus_m4_linear_param_indices(hamiltonian, *args, **kwargs))
        else:
            magnus = lambda *args, **kwargs: magnus_m6_linear(hamiltonian, *args, **kwargs)
            magnus_param_indices = (lambda *args, **kwargs:
                                    magnus_m6_linear_param_indices(hamiltonian, *args, **kwargs))
    #ENDIF
    
    magnus_wrapper = lambda *args, **kwargs: anp.real(magnus(*args, **kwargs))
    def dmagnus_dparams(*args, **kwargs):
        _magnus, _dmagnus_dparams = ans_jacobian(magnus_wrapper, 1)(*args, **kwargs)
        old_axes = np.arange(len(_dmagnus_dparams.shape))
        new_axes = np.roll(old_axes, -2)
        return (np.moveaxis(_dmagnus_dparams, old_axes, new_axes),
                _magnus)

    return dmagnus_dparams, magnus, magnus_param_indices
