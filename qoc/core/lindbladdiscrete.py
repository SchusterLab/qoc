"""
linbladdiscrete.py - a module to evolve a set of density matrices
under the lindblad master equation using time-discrete
control parameters
"""

import numpy as np

from qoc.models import (EvolveLindbladDiscreteState,
                        EvolveLindbladResult,
                        InterpolationPolicy,
                        OperationPolicy,
                        evolve_step_lindblad_discrete)

### MAIN METHODS ###

def evolve_lindblad_discrete(control_step_count, evolution_time,
                             initial_densities,
                             controls=None, costs=list(),
                             hamiltonian=None,
                             interpolation_policy=InterpolationPolicy.LINEAR,
                             lindblad_data=None,
                             operation_policy=OperationPolicy.CPU,
                             system_step_multiplier=1,):
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
    lindblad_data :: (time :: float) -> (tuple(operators :: ndarray, dissipators :: ndarray))
        - a function to generate the dissipation constants and lindblad operators for a given time,
          an array of operators should be returned even if there 
          are zero or one dissipator and operator pairs
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    system_step_multiplier :: int - the multiple of control_step_count at which
        the system should evolve, control parameters will be interpolated at
        these steps

    Returns:
    result :: qoc.models.EvolveLindbladResult - information
        about the evolution
    """
    pstate = EvolveLindbladDiscreteState(control_step_count, controls,
                                         costs,
                                         evolution_time,
                                         hamiltonian, initial_densities,
                                         interpolation_policy,
                                         lindblad_data,
                                         operation_policy,
                                         system_step_multiplier,)
    result = EvolveLindbladResult()

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
    hamiltonian = pstate.hamiltonian
    final_control_step = pstate.final_control_step
    final_system_step = pstate.final_system_step
    interpolation_policy = pstate.interpolation_policy
    lindblad_data = pstate.lindblad_data
    operation_policy = pstate.operation_policy
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0

    for system_step in range(final_system_step + 1):
        control_step, _ = divmod(system_step, system_step_multiplier)
        is_final_control_step = control_step == final_control_step
        is_final_system_step = system_step == final_system_step
        time = system_step * dt

        # Evolve the density matrices.
        densities = evolve_step_lindblad_discrete(densities, dt,
                                                  time, is_final_control_step, control_step,
                                                  controls, hamiltonian,
                                                  interpolation_policy,
                                                  lindblad_data,
                                                  operation_policy)

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

def _test():
    import numpy as np
    from qutip import mesolve, Qobj, Options
    
    from qoc.standard import (conjugate_transpose,
                              PAULI_X, PAULI_Y,
                              matrix_to_column_vector_list,
                              SIGMA_PLUS, SIGMA_MINUS,)

    _BIG = int(1e1)
    
    def _generate_complex_matrix(matrix_size):
        return (np.random.rand(matrix_size, matrix_size)
                + 1j * np.random.rand(matrix_size, matrix_size))
    
    def _generate_hermitian_matrix(matrix_size):
        matrix = _generate_complex_matrix(matrix_size)
        return (matrix + conjugate_transpose(matrix)) * 0.5

    # Test that evolution WITH a hamiltonian and WITHOUT lindblad operators
    # yields a known result.
    # Use e.q. 109 from
    # https://arxiv.org/pdf/1904.06560.pdf.
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    iswap_unitary = np.array(((1,   0,   0, 0),
                              (0,   0, -1j, 0),
                              (0, -1j,   0, 0),
                              (0,   0,   0, 1)))
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(iswap_unitary)
    initial_densities = np.matmul(initial_states, conjugate_transpose(initial_states))
    target_densities = np.matmul(target_states, conjugate_transpose(target_states))
    system_hamiltonian = ((1/ 2) * (np.kron(PAULI_X, PAULI_X)
                              + np.kron(PAULI_Y, PAULI_Y)))
    hamiltonian = lambda controls, time: system_hamiltonian
    control_step_count = int(1e3)
    evolution_time = np.pi / 2
    result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                      initial_densities, hamiltonian=hamiltonian)
    final_densities = result.final_densities
    assert(np.allclose(final_densities, target_densities))
    # Note that qutip only gets this result within 1e-5 error.
    tlist = np.linspace(0, evolution_time, control_step_count)
    c_ops = list()
    e_ops = list()
    options = Options(nsteps=control_step_count)
    for i, initial_density in enumerate(initial_densities):
        result = mesolve(Qobj(system_hamiltonian),
                         Qobj(initial_density),
                         tlist, c_ops, e_ops,
                         options=options)
        final_density = result.states[-1].full()
        target_density = target_densities[i]
        assert(np.allclose(final_density, target_density, atol=1e-5))
    #ENDFOR

    # Test that evolution WITHOUT a hamiltonian and WITH lindblad operators
    # yields a known result.
    # This test ensures that dissipators are working correctly.
    # Use e.q.14 from
    # https://inst.eecs.berkeley.edu/~cs191/fa14/lectures/lecture15.pdf.
    hilbert_size = 2
    gamma = 2
    lindblad_dissipators = np.array((gamma,))
    lindblad_operators = np.stack((SIGMA_MINUS,))
    lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
    evolution_time = 1.
    control_step_count = int(1e3)
    inv_sqrt_2 = 1 / np.sqrt(2)
    a0 = np.random.rand()
    c0 = 1 - a0
    b0 = np.random.rand()
    b0_star = np.conj(b0)
    initial_density_0 = np.array(((a0,        b0),
                                  (b0_star,   c0)))
    initial_densities = np.stack((initial_density_0,))
    gt = gamma * evolution_time
    expected_final_density = np.array(((1 - c0 * np.exp(- gt),    b0 * np.exp(-gt/2)),
                                       (b0_star * np.exp(-gt/2), c0 * np.exp(-gt))))
    result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                      initial_densities,
                                      lindblad_data=lindblad_data)
    final_density = result.final_densities[0]
    assert(np.allclose(final_density, expected_final_density))

    # Test that evolution WITH a random hamiltonian and WITH random lindblad operators
    # yields a similar result to qutip.
    # Note that the allclose tolerance may need to be adjusted.
    for matrix_size in range(2, _BIG):
        # Evolve under lindbladian.
        hamiltonian_matrix = _generate_hermitian_matrix(matrix_size)
        hamiltonian = lambda controls, time: hamiltonian_matrix
        lindblad_operator_count = np.random.randint(1, matrix_size)
        lindblad_operators = np.stack([_generate_complex_matrix(matrix_size)
                                      for _ in range(lindblad_operator_count)])
        lindblad_dissipators = np.ones((lindblad_operator_count))
        lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
        density_matrix = _generate_hermitian_matrix(matrix_size)
        initial_densities = np.stack((density_matrix,))
        evolution_time = 1
        control_step_count = int(1e4)
        result = evolve_lindblad_discrete(control_step_count, evolution_time,
                                          initial_densities,
                                          hamiltonian=hamiltonian,
                                          lindblad_data=lindblad_data)
        final_density = result.final_densities[0]

        # Evolve under lindbladian with qutip.
        hamiltonian_qutip =  Qobj(hamiltonian_matrix)
        initial_density_qutip = Qobj(density_matrix)
        lindblad_operators_qutip = [Qobj(lindblad_operator)
                                    for lindblad_operator in lindblad_operators]
        e_ops_qutip = list()
        tlist = np.linspace(0, evolution_time, control_step_count)
        options = Options(nsteps=control_step_count)
        result_qutip = mesolve(hamiltonian_qutip,
                               initial_density_qutip,
                               tlist,
                               lindblad_operators_qutip,
                               e_ops_qutip,)
        final_density_qutip = result_qutip.states[-1].full()
        
        assert(np.allclose(final_density, final_density_qutip,))
    #ENDFOR


if __name__ == "__main__":
    _test()
