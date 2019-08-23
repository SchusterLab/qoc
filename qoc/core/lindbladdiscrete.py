"""
linbladdiscrete.py - a module to evolve a set of density matrices
under the lindblad master equation using time-discrete
control parameters
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import (EvolveLindbladDiscreteState,
                        EvolveLindbladDiscreteResult,
                        InterpolationPolicy, OperationPolicy)

### MAIN METHODS ###

def grape_lindblad_discrete():
    pass


def evolve_lindblad_discrete(control_step_count, evolution_time, hamiltonian,
                             initial_densities, lindblad_operators,
                             controls=None, costs=None,                        
                             interpolation_policy=InterpolationPolicy.LINEAR,
                             operation_policy=OperationPolicy.CPU,
                             system_step_multiplier=1.,):
    """
    Evolve a set of density matrices under the lindblad equation
    and compute the optimization error.

    Args:
    control_step_count :: int - the number of time intervals in the 
        evolution time in which the controls are spaced, or, if no controls
        are specified, the number of time steps in which the evolution time
        interval should be broken up
    controls :: ndarray - the controls that should be provided to the
        hamiltonian for the evolution
    costs :: iterable(qoc.models.Cost) - the cost functions to guide
        optimization
    evolution_time :: float - the length of time the system should evolve for
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_operators :: (time :: float) -> operators :: ndarray
        - a function to generate the lindblad operators for a given time,
          an array of operators should be returned even if there is only
          one operator
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    system_step_multiplier :: int - the multiple of control_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps

    Returns:
    result :: qoc.models.EvolveLindbladDiscreteResult - information
        about the evolution
    """
    pstate = EvolveLindbladDiscreteState(control_step_count, controls,
                                         costs, evolution_time,
                                         hamiltonian, initial_densities
                                         interpolation_policy,
                                         lindblad_operators,
                                         operation_policy,
                                         system_step_multiplier)
    result = EvolveLindbladDiscreteResult

    _ = _evaluate_lindblad(None, pstate, result)

    return result


### HELPER METHODS ###

def _evaluate_lindblad(controls, pstate, reporter):
    """
    Evolve a set of density matrices under the lindblad equation
    and compute associated optimization costs.

    Args:
    controls :: ndarray - the control parameters
    pstate :: qoc.models.GrapeLindbladDiscreteState
        or qoc.models.EvolveLindbladDiscreteState - the program state
    reporter :: any - the object to keep track of relevant information

    Returns:
    total_error :: float - the optimization cost for the provided controls
    """
    # Initialize local variables (heap -> stack).
    costs = pstate.costs
    densities = pstate.initial_densities
    dt = pstate.dt
    evolve_lindblad = pstate.evolve_lindblad
    final_control_step = pstate.final_control_step
    final_system_step = pstate.final_system_step
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0

    for system_step in range(final_system_step + 1):
        control_step, _ = anp.divmod(system_step, system_step_multiplier)
        is_final_control_step = control_step == final_control_step
        is_final_system_step = system_step == final_system_step
        time = system_step * dt

        # Evolve the density matrices.
        densities = evolve_lindblad(controls, control_step, densities, dt,
                                    time, is_final_control_step)

        # Compute the costs.
        if is_final_system_step:
            for i, cost in enumerate(costs):
                error = cost.cost(controls, densities, system_step)
                total_error = total_error + error
            #ENDFOR
            reporter.final_densities = densities
            reporter.total_error = total_error
        else:
            for i, step_cost in enumerate(step_costs):
                error = step_cost.cost(controls, densities, system_step)
                total_error = total_error + error
            #ENDFOR
    #ENDFOR

    return total_error


### MODULE TESTS ###

_BIG = 100

def _generate_complex_matrix(matrix_size):
    return (np.random.rand(matrix_size, matrix_size)
            + 1j * np.random.rand(matrix_size, matrix_size))

def _test():
    from qutip import mesolve, Qobj
    from qoc.standard import conjugate_transpose
        
    # Test that lindblad evolution yields similar result to qutip.    
    for matrix_size in range(2, _BIG):
        # Generate necessary evolution objects.
        matrix = _generate_complex_matrix(matrix_size)
        hamiltonian_matrix = (matrix + conjugate_transpose(matrix)) / 2
        hamiltonian = lambda controls, time: hamiltonian_matrix
        lindblad_matrices = [_generate_complex_matrix(matrix_size)
                                      for _ in np.arange(np.random.randint(0, matrix_size))]
        if len(lindblad_matrices) != 0:
            lindblad_matrices = np.stack(lindblad_matrices)
            lindblad_operators = lambda time: lindblad_matrices
        else:
            zero_matrix = np.zeros((matrix_size, matrix_size))
            lindblad_operators = lambda time: zero_matrix
        density_matrix = _generate_complex_matrix(matrix_size)
        initial_densities = np.stack((density_matrix,))
        evolution_time = control_step_count = _BIG

        # Generate qutip compatible objects.
        qutip_expectation_operators = []
        qutip_hamiltonian = Qobj(hamiltonian_matrix)
        qutip_initial_density = Qobj(density_matrix)
        qutip_lindblad_operators = [Qobj(lindblad_matrix)
                                    for lindblad_matrix in lindblad_matrices]
        qutip_tlist = np.linspace(0, evolution_time, control_step_count + 1)
        
        eld_result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                              hamiltonian, initial_densities,
                                              lindblad_operators)
        eld_final_density_matrix = eld_result.final_densities[0]
        print("eld_final_density_matrix:\n{}"
              "".format(eld_final_density_matrix))
        
        qutip_result = mesolve(qutip_hamiltonian, qutip_initial_density,
                                     qutip_tlist, qutip_lindblad_operators,
                                     qutip_expectation_operators)
        qutip_final_density_matrix = qutip_result.states[-1]
        print("qutip_final_density_matrix:\n{}"
              "".format(qutip_final_density_matrix))
        
        # assert(np.allclose(eld_final_density_matrix, qutip_final_density_matrix))
    #ENDFOR


if __name__ == "__main__":
    _test()
