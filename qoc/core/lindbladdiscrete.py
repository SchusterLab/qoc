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
"""
lindbladmethods.py - a module for lindblad math

NOTE:
This module's methods are tested by the
tests on the evolve_lindblad_discrete
method in qoc.core.gld.py.
"""

from qoc.models.mathmethods import interpolate_linear
from qoc.models.interpolationpolicy import (InterpolationPolicy,)
from qoc.models.operationpolicy import (OperationPolicy,)
from qoc.standard import (commutator, conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def evolve_step_lindblad_discrete(densities, dt,
                                  time, control_sentinel=False,
                                  control_step=0, 
                                  controls=None,
                                  hamiltonian=None,
                                  interpolation_policy=InterpolationPolicy.LINEAR,
                                  lindblad_data=None,
                                  operation_policy=OperationPolicy.CPU,):
    """
    Use Runge-Kutta 4th order to evolve the density matrices to the next time step
    under the lindblad master equation. This RK4 implementation follows the definition:
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods. Runge-Kutta was
    chosen over matrix exponential by the suggestion of: https://arxiv.org/abs/1609.03170.
    

    NOTATION:
     - t is time, c is controls, h is hamiltonian, g is dissipation constants,
       l is lindblad operators, k are the runge-kutta increments

    Args:
    control_sentinel :: bool - set to True if this is the final control step,
        in which case control interpolation is performed on the last two
        control sets in the controls array
    control_step :: int - the index into the control array at which control
        interpolation should be performed
    controls :: ndarray - the controls that should be provided to the
        hamiltonian for the evolution    
    densities :: ndarray - the probability density matrices to evolve
    dt :: float - the time step
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_data :: (time :: float) -> (dissipartors :: ndarray, operators :: ndarray)
        - a function to generate the dissipation constants and lindblad operators
          for a given time
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    time :: float - the current evolution time

    Returns:
    densities :: ndarray - the densities evolved to `time + dt`
    """
    t1 = time
    t2 = time + 0.5 * dt
    t3 = t2
    t4 = time + dt
    if controls is None:
        c1 = c2 = c3 = c4 = None
    else:
        if control_sentinel:
            control_left = controls[control_step - 1]
            control_right = controls[control_step]
        else:
            control_left = controls[control_step]
            control_right = controls[control_step + 1]
        c1 = control_left
        if interpolation_policy == InterpolationPolicy.LINEAR:
            c2 = interpolate_linear(t1, t4, t2, control_left, control_right)
        else:
            raise ValueError("The interpolation policy {} is not "
                             "implemented for this method."
                             "".format(interpolation_policy))
        c3 = c2
        c4 = control_right
    #ENDIF
    if hamiltonian is None:
        h1 = h2 = h3 = h4 = None
    else:
        h1 = hamiltonian(c1, t1)
        h2 = hamiltonian(c2, t2)
        h3 = h2
        h4 = hamiltonian(c4, t4)
    if lindblad_data is None:
        (g1, l1) =  (g2, l2) = (g3, l3) = (g4, l4) = (None, None)
    else:
        g1, l1 = lindblad_data(t1)
        g2, l2 = lindblad_data(t2)
        g3, l3 = lindblad_data(t3)
        g4, l4 = lindblad_data(t4)
    #ENDIF
    k1 = dt * get_lindbladian(densities, g1, h1, l1,
                              operation_policy=operation_policy)
    k2 = dt * get_lindbladian(densities + 0.5 * k1, g2, h2, l2,
                              operation_policy=operation_policy)
    k3 = dt * get_lindbladian(densities + 0.5 * k2, g3, h3, l3,
                              operation_policy=operation_policy)
    k4 = dt * get_lindbladian(densities + k3, g4, h4, l4,
                              operation_policy=operation_policy)

    densities = densities + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return densities


def get_lindbladian(densities, dissipators=None, hamiltonian=None,
                    operators=None,
                    operation_policy=OperationPolicy.CPU):
    """
    Compute the action of the lindblad operator on a single (set of)
    density matrix (matrices). This implementation uses the definiton:
    https://en.wikipedia.org/wiki/Lindbladian.

    Args:
    densities :: ndarray - the probability density matrices
    dissipators :: ndarray - the lindblad dissipators
    hamiltonian :: ndarray
    operators :: ndarray - the lindblad operators
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.

    Returns:
    lindbladian :: ndarray - the lindbladian operator acting on the densities
    """
    if not (hamiltonian is None):
        lindbladian = -1j * commutator(hamiltonian, densities,
                                       operation_policy=operation_policy)
    else:
        lindbladian = 0
    if ((not (operators is None))
      and (not (dissipators is None))):
        operators_dagger = conjugate_transpose(operators,
                                               operation_policy=operation_policy)
        operators_product = matmuls(operators_dagger, operators,
                                    operation_policy=operation_policy)
        for i, operator in enumerate(operators):
            dissipator = dissipators[i]
            operator_dagger = operators_dagger[i]
            operator_product = operators_product[i]
            lindbladian = (lindbladian
                           + (dissipator
                              * (matmuls(operator, densities, operator_dagger,
                                         operation_policy=operation_policy)
                                 - 0.5 * matmuls(operator_product, densities,
                                                 operation_policy=operation_policy)
                                 - 0.5 * matmuls(densities, operator_product,
                                                 operation_policy=operation_policy))))
        #ENDFOR
    #ENDIF
    return lindbladian


### MODULE TESTS ###

def _test():
    import numpy as np
    
    # Test get_lindbladian on a hand verified solution.
    p = np.array(((1, 1), (1, 1)))
    ps = np.stack((p,))
    h = np.array(((0, 1), (1, 0)))
    g = 1
    gs = np.array((1,))
    l = np.array(((1, 0), (0, 0)))
    ls = np.stack((l,))
    lindbladian = get_lindbladian(p, gs, h, ls)
    expected_lindbladian = np.array(((0, -0.5),
                                     (-0.5, 0)))
    assert(np.allclose(lindbladian, expected_lindbladian))

if __name__ == "__main__":
    _test()
