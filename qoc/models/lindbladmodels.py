"""
lindbladmodels.py - a module to define classes to
encapsulate the necessary information to execute 
programs involving lindblad evolution
"""

from .programstate import (ProgramState, GrapeState)

### MAIN STRUCTURES ###

class EvolveLindbladDiscreteState(ProgramState):
    """
    This class encapsulates the necessary information to evolve
    a set of density matrices under the lindblad equation and compute
    optimization error for one round.

    Fields:
    costs :: iterable(qoc.models.Cost) - the cost functions that
        define the cost model for evolution
    dt :: float - the time step used for evolution, it is the time
        inbetween system steps
    final_control_step :: int - the ultimate index into the control array
    final_system_step :: int - the last time step
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    initial_densities :: ndarray - the probability density matrices
        to evolve
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_data :: (time :: float) -> (dissapartors :: ndarray, operators :: ndarray)
        - a function to generate the dissapation constants and lindblad operators
          for a given time,
          an array of dissipators and operators should be returned even if there
          are zero or one dissapator and operator pairs
    operation_policy :: qoc.models.OperationPolicy - defines how
        computations should be performed, e.g. CPU, GPU, sparse, etc.
    step_costs :: iterable(qoc.models.Cost) - the cost functions that
        define the cost model for evolution that should be evaluated
        at every time step
    step_cost_indices :: iterable(int)- a list of the indices in the costs
        array which are step costs
    system_step_multiplier :: int - this value times `control_step_count`
        determines how many time steps are used in evolution
    """
    
    def __init__(self, control_step_count,
                 controls, costs, evolution_time,
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
        controls :: ndarray - the control parameters to feed to the hamiltonian
        evolution_time :: float - the time over which the system will evolve
        """
        super().__init__(control_step_count, controls, costs,
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
