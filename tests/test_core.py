"""
test_core.py - This module provides unit tests on the qoc.core module.
"""
### qoc.core.common.py ###

def test_clip_control_norms():
    import numpy as np
    from qoc.core.common import clip_control_norms

    # Control norm clipping.
    controls = np.array(((1+2j, 7+8j), (3+4j, 5), (5+6j, 10,), (1-3j, -10),))
    max_control_norms = np.array((7, 8,))
    expected_clipped_controls = np.array(((1+2j, (7+8j) * np.divide(8, np.sqrt(113))),
                                          (3+4j, 5),
                                          ((5+6j) * np.divide(7, np.sqrt(61)), 8,),
                                          (1-3j, -8)))
    clip_control_norms(controls, max_control_norms)
    
    assert(np.allclose(controls, expected_clipped_controls))


def test_strip_slap():
    import numpy as np
    from qoc.models.dummy import Dummy
    from qoc.core.common import (slap_controls, strip_controls,)

    big = 100
    pstate = Dummy()
    pstate.complex_controls = True
    shape_range = np.arange(big) + 1
    for step_count in shape_range:
        for control_count in shape_range:
            pstate.controls_shape = controls_shape = (step_count, control_count)
            pstate.max_control_norms = np.ones(control_count) * 2
            controls = np.random.rand(*controls_shape) + 1j * np.random.rand(*controls_shape)
            stripped_controls = strip_controls(pstate, controls)
            assert(stripped_controls.ndim == 1)
            assert(not (stripped_controls.dtype in (np.complex64, np.complex128)))
            transformed_controls = slap_controls(pstate.complex_controls, stripped_controls,
                                                 pstate.controls_shape)
            assert(np.allclose(controls, transformed_controls))
            assert(controls.shape == transformed_controls.shape)
        #ENDFOR
    #ENDFOR

    pstate.complex_controls = False
    for step_count in shape_range:
        for control_count in shape_range:
            pstate.controls_shape = controls_shape = (step_count, control_count)
            pstate.max_control_norms = np.ones(control_count)
            controls = np.random.rand(*controls_shape)
            stripped_controls = strip_controls(pstate.complex_controls, controls)
            assert(stripped_controls.ndim == 1)
            assert(not (stripped_controls.dtype in (np.complex64, np.complex128)))
            transformed_controls = slap_controls(pstate.complex_controls, stripped_controls,
                                                 pstate.controls_shape)
            assert(np.allclose(controls, transformed_controls))
            assert(controls.shape == transformed_controls.shape)
        #ENDFOR
    #ENDFOR


### qoc.core.lindbladdiscrete.py ###

def test_evolve_lindblad_discrete():
    """
    Run end-to-end tests on evolve_lindblad_discrete.
    """
    import numpy as np
    from qutip import mesolve, Qobj

    from qoc.core.lindbladdiscrete import evolve_lindblad_discrete
    from qoc.standard import (conjugate_transpose,
                              SIGMA_X, SIGMA_Y,
                              matrix_to_column_vector_list,
                              SIGMA_PLUS, SIGMA_MINUS,)

    big = 4
    
    # Test that evolution WITH a hamiltonian and WITHOUT lindblad operators
    # yields a known result.
    # Use e.q. 109 from
    # https://arxiv.org/abs/1904.06560
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
    system_hamiltonian = ((1/ 2) * (np.kron(SIGMA_X, SIGMA_X)
                              + np.kron(SIGMA_Y, SIGMA_Y)))
    hamiltonian = lambda controls, time: system_hamiltonian
    system_eval_count = 2
    evolution_time = np.pi / 2
    result = evolve_lindblad_discrete(evolution_time,
                                      initial_densities,
                                      system_eval_count,
                                      hamiltonian=hamiltonian)
    final_densities = result.final_densities
    assert(np.allclose(final_densities, target_densities))
    # Note that qutip only gets this result within 1e-5 error.
    tlist = np.array([0, evolution_time])
    c_ops = list()
    e_ops = list()
    for i, initial_density in enumerate(initial_densities):
        result = mesolve(Qobj(system_hamiltonian),
                         Qobj(initial_density),
                         tlist, c_ops, e_ops,)
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
    sigma_plus = np.array([[0, 1], [0, 0]])
    lindblad_operators = np.stack((sigma_plus,))
    lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
    evolution_time = 1.
    system_eval_count = 2
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
    result = evolve_lindblad_discrete(evolution_time,
                                      initial_densities,
                                      system_eval_count,
                                      lindblad_data=lindblad_data)
    final_density = result.final_densities[0]
    assert(np.allclose(final_density, expected_final_density))

    # Test that evolution WITH a random hamiltonian and WITH random lindblad operators
    # yields a similar result to qutip.
    # Note that the allclose tolerance may need to be adjusted.
    matrix_size = 4
    for i in range(big):
        # Evolve under lindbladian.
        hamiltonian_matrix = random_hermitian_matrix(matrix_size)
        hamiltonian = lambda controls, time: hamiltonian_matrix
        lindblad_operator_count = np.random.randint(1, matrix_size)
        lindblad_operators = np.stack([random_complex_matrix(matrix_size)
                                      for _ in range(lindblad_operator_count)])
        lindblad_dissipators = np.ones((lindblad_operator_count,))
        lindblad_data = lambda time: (lindblad_dissipators, lindblad_operators)
        density_matrix = random_hermitian_matrix(matrix_size)
        initial_densities = np.stack((density_matrix,))
        evolution_time = 5
        system_eval_count = 2
        result = evolve_lindblad_discrete(evolution_time,
                                          initial_densities,
                                          system_eval_count,
                                          hamiltonian=hamiltonian,
                                          lindblad_data=lindblad_data)
        final_density = result.final_densities[0]

        # Evolve under lindbladian with qutip.
        hamiltonian_qutip =  Qobj(hamiltonian_matrix)
        initial_density_qutip = Qobj(density_matrix)
        lindblad_operators_qutip = [Qobj(lindblad_operator)
                                    for lindblad_operator in lindblad_operators]
        e_ops_qutip = list()
        tlist = np.array((0, evolution_time,))
        result_qutip = mesolve(hamiltonian_qutip,
                               initial_density_qutip,
                               tlist,
                               lindblad_operators_qutip,
                               e_ops_qutip,)
        final_density_qutip = result_qutip.states[-1].full()
        assert(np.allclose(final_density, final_density_qutip))
    #ENDFOR


def test_grape_lindblad_discrete():
    """
    Run end-to-end test on the grape_lindblad_discrete function.

    NOTE: We mostly care about the tests for evolve_lindblad_discrete.
    For grape_lindblad_discrete we care that everything is being passed
    through functions properly, but autograd has a really solid testing
    suite and we trust that their gradients are being computed
    correctly.
    """
    import numpy as np

    from qoc.core.lindbladdiscrete import grape_lindblad_discrete
    from qoc.standard import (conjugate_transpose,
                              ForbidDensities, SIGMA_X, SIGMA_Y,)
    
    # Test that parameters are clipped if they grow too large.
    hilbert_size = 4
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(SIGMA_X, SIGMA_X)
                                            + np.kron(SIGMA_Y, SIGMA_Y))
    hamiltonian = lambda controls, t: (controls[0] * hamiltonian_matrix)
    initial_states = np.array([[[0], [1], [0], [0]]])
    initial_densities = np.matmul(initial_states, conjugate_transpose(initial_states))
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    forbidden_densities = np.matmul(forbidden_states, conjugate_transpose(forbidden_states))
    control_count = 1
    evolution_time = 10
    system_eval_count = control_eval_count = 11
    max_norm = 1e-10
    max_control_norms = np.repeat(max_norm, control_count)
    costs = [ForbidDensities(forbidden_densities, system_eval_count)]
    iteration_count = 5
    log_iteration_step = 0
    result = grape_lindblad_discrete(control_count, control_eval_count,
                                     costs, evolution_time,
                                     initial_densities,
                                     system_eval_count,
                                     hamiltonian=hamiltonian,
                                     iteration_count=iteration_count,
                                     log_iteration_step=log_iteration_step,
                                     max_control_norms=max_control_norms)
    for i in range(result.best_controls.shape[1]):
        assert(np.less_equal(np.abs(result.best_controls[:,i]),
                             max_control_norms[i]).all())


### qoc.core.mathmethods.py ###

def test_get_lindbladian():
    import numpy as np
    from qoc.core.mathmethods import get_lindbladian
    
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


def test_interpolate_linear_points():
    import numpy as np
    from qoc.core.mathmethods import interpolate_linear_points
    
    big = int(1e3)
    big_div_2 = big / 2
    
    for i in range(big):
        # Generate a line with a constant slope between -5 and 5.
        line = lambda x: slope * x
        slope = np.random.rand() * 10 - 5
        x1 = np.random.rand() * big - big_div_2
        x2 = np.random.rand() * big - big_div_2
        x3 = np.random.rand() * big - big_div_2
        # Check that the linear method approximates the line
        # exactly.
        y1 = line(x1)
        y2 = line(x2)
        y3 = line(x3)
        y3_interpolated = interpolate_linear_points(x1, x2, x3, y1, y2)
        assert(np.isclose(y3_interpolated, y3))
    #ENDFOR


# TODO(tpr0p): Rewrite this test for time dependent matrices.
def test_magnus():
    import numpy as np
    from qoc.core.mathmethods import (magnus_m2, magnus_m4, magnus_m6,)
    
    # These tests ensure the above methods were copied to code correclty.
    # They are hand checked. There may be a better way to test the methods.
    dt = 1.
    identity = np.eye(2)
    # assert(np.allclose(magnus_m2(identity, dt), identity))
    # assert(np.allclose(magnus_m4(identity, identity, dt), identity))
    # assert(np.allclose(magnus_m6(identity, identity, identity, dt), identity))
    dt = 2.
    a1 = np.array([[2., 3.], [4., 5.]])
    a2 = np.array([[9., 6.], [8., 7.]])
    a3 = np.array([[12., 13.], [11., 10.]])
    # assert(np.allclose(magnus_m2(a1, dt),
    #                   np.array([[4., 6.],
    #                             [8., 10.]])))
    # assert(np.allclose(magnus_m4(a1, a2, dt),
    #                   np.array([[11., 22.85640646],
    #                             [-6.47520861, 12.]])))
    # assert(np.allclose(magnus_m6(a1, a2, a3, dt),
    #                   np.array([[-241.71158615, 100.47657236],
    #                             [310.29160996, 263.71158615]])))


def test_rkdp5():
    """
    Test rkdp5 using an ode with a known solution.

    References:
    [1] http://tutorial.math.lamar.edu/Classes/DE/Exact.aspx
    """
    import numpy as np
    from qoc.core.mathmethods import integrate_rkdp5
    
    COMPARE = False

    # Problem setup.
    x0 = 0
    x1 = 10
    y0 = np.array((-3,))
    y_sol = lambda x: 0.5 * (-(x ** 2 + 1) - (np.sqrt(x ** 4 + 12 * x ** 3 + 2 * x ** 2 + 25)))
    def rhs(x, y):
        return ((-2 * x * y + 9 * x ** 2) / (2 * y + x ** 2 + 1))

    # Analytical solution.
    y_1_expected = y_sol(x1)

    # QOC solution.
    y_1 = integrate_rkdp5(rhs, np.array([x1]), x0, y0)[0]

    assert(np.allclose(y_1, y_1_expected))

    if COMPARE:
        from scipy.integrate import ode, solve_ivp

        # Scipy fortran solutions.
        r = ode(rhs).set_integrator("vode", method="bdf")
        r.set_initial_value(y0, x0)
        y_1_scipy_vode = r.integrate(x1)[0]

        r = ode(rhs).set_integrator("dopri5")
        r.set_initial_value(y0, x0)
        y_1_scipy_dopri5_f = r.integrate(x1)[0]

        # Scipy python solutions.
        res = solve_ivp(rhs, [x0, x1], y0, method="RK45")
        y_1_scipy_dopri5_py = res.y[:, -1][0]

        res = solve_ivp(rhs, [x0, x1], y0, method="Radau")
        y_1_scipy_radau_py = res.y[:, -1][0]

        print("y_1_expected:\n{}"
              "".format(y_1_expected))
        print("y_1_scipy_vode:\n{}"
              "".format(y_1_scipy_vode))
        print("y_1_scipy_dopri5_f:\n{}"
              "".format(y_1_scipy_dopri5_f))
        print("y_1_scipy_dopri5_py:\n{}"
              "".format(y_1_scipy_dopri5_py))
        print("y_1_qoc:\n{}"
              "".format(y_1))


### qoc.core.schroedingerdiscrete.py ###

def test_evolve_schroedinger_discrete():
    """
    Run end-to-end test on the evolve_schroedinger_discrete
    function.
    """
    import numpy as np
    from qutip import mesolve, Qobj, Options

    from qoc.core import evolve_schroedinger_discrete
    from qoc.models import (MagnusPolicy,)
    from qoc.standard import (matrix_to_column_vector_list,
                              SIGMA_X, SIGMA_Y,)

    big = 10
    magnus_policies = (MagnusPolicy.M2, MagnusPolicy.M4, MagnusPolicy.M6,)

    # Test that evolving states under a hamiltonian yields
    # a known result. Use e.q. 109 of 
    # https://arxiv.org/abs/1904.06560.
    hilbert_size = 4
    identity_matrix = np.eye(hilbert_size, dtype=np.complex128)
    iswap_unitary = np.array(((1,   0,   0, 0),
                              (0,   0, -1j, 0),
                              (0, -1j,   0, 0),
                              (0,   0,   0, 1)))
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(SIGMA_X, SIGMA_X)
                                     + np.kron(SIGMA_Y, SIGMA_Y))
    hamiltonian = lambda controls, time: hamiltonian_matrix
    initial_states = matrix_to_column_vector_list(identity_matrix)
    target_states = matrix_to_column_vector_list(iswap_unitary)
    evolution_time = np.divide(np.pi, 2)
    system_eval_count = int(1e3)
    for magnus_policy in magnus_policies:
        result = evolve_schroedinger_discrete(evolution_time, hamiltonian,
                                              initial_states, system_eval_count,
                                              magnus_policy=magnus_policy)
        final_states = result.final_states
        assert(np.allclose(final_states, target_states))
    #ENDFOR
    # Note that qutip only gets the same result within 1e-6 error.
    tlist = np.array([0, evolution_time])
    c_ops = list()
    e_ops = list()
    hamiltonian_qutip = Qobj(hamiltonian_matrix)
    for i, initial_state in enumerate(initial_states):
        initial_state_qutip = Qobj(initial_state)
        result = mesolve(hamiltonian_qutip,
                         initial_state_qutip,
                         tlist, c_ops, e_ops,)
        final_state = result.states[-1].full()
        target_state = target_states[i]
        assert(np.allclose(final_state, target_state, atol=1e-6))
    #ENDFOR

    # Test that evolving states under a random hamiltonian yields
    # a result similar to qutip.
    hilbert_size = 4
    initial_state = np.divide(np.ones((hilbert_size, 1)),
                              np.sqrt(hilbert_size))
    initial_states = np.stack((initial_state,))
    initial_state_qutip = Qobj(initial_state)
    system_eval_count = int(1e3)
    evolution_time = 1
    tlist = np.array([0, evolution_time])
    c_ops = e_ops = list()
    for _ in range(big):
        hamiltonian_matrix = random_hermitian_matrix(hilbert_size)
        hamiltonian = lambda controls, time: hamiltonian_matrix
        hamiltonian_qutip = Qobj(hamiltonian_matrix)
        result = mesolve(hamiltonian_qutip,
                         initial_state_qutip,
                         tlist, c_ops, e_ops,)
        final_state_qutip = result.states[-1].full()
        for magnus_policy in magnus_policies:
            result = evolve_schroedinger_discrete(evolution_time, hamiltonian,
                                                  initial_states, system_eval_count,
                                                  magnus_policy=magnus_policy)
            final_state = result.final_states[0]
            assert(np.allclose(final_state, final_state_qutip, atol=1e-4))
        #ENDFOR
    #ENDFOR
        

def test_grape_schroedinger_discrete():
    """
    Run end-to-end test on the grape_schroedinger_discrete function.

    NOTE: We mostly care about the tests for evolve_schroedinger_discrete.
    For grape_schroedinger_discrete, we care that everything is being passed
    through functions properly. Autograd has a really solid testing
    suite; we trust that their gradients are being computed
    correctly.
    """
    import numpy as np

    from qoc.core import grape_schroedinger_discrete
    from qoc.standard import (ForbidStates, SIGMA_X, SIGMA_Y,)
    
    # Test that parameters are clipped if they grow too large.
    hilbert_size = 4
    hamiltonian_matrix = np.divide(1, 2) * (np.kron(SIGMA_X, SIGMA_X)
                                            + np.kron(SIGMA_Y, SIGMA_Y))
    hamiltonian = lambda controls, t: (controls[0] * hamiltonian_matrix)
    initial_states = np.array([[[0], [1], [0], [0]]])
    forbidden_states = np.array([[[[0], [1], [0], [0]]]])
    control_count = 1
    evolution_time = 10
    control_eval_count = system_eval_count = 11
    max_norm = 1e-10
    max_control_norms = np.repeat(max_norm, control_count)
    costs = [ForbidStates(forbidden_states, system_eval_count)]
    iteration_count = 100
    log_iteration_step = 0
    result = grape_schroedinger_discrete(control_count, control_eval_count,
                                         costs, evolution_time,
                                         hamiltonian, initial_states,
                                         system_eval_count,
                                         iteration_count=iteration_count,
                                         log_iteration_step=log_iteration_step,
                                         max_control_norms=max_control_norms)
    for i in range(result.best_controls.shape[1]):
        assert(np.less_equal(np.abs(result.best_controls[:,i]),
                             max_control_norms[i]).all())


### utility methods ###

def random_complex_matrix(matrix_size):
    """
    Generate a random, square, complex matrix of size `matrix_size`.
    """
    import numpy as np
    return (np.random.rand(matrix_size, matrix_size)
            + 1j * np.random.rand(matrix_size, matrix_size))


def random_hermitian_matrix(matrix_size):
    """
    Generate a random, square, hermitian matrix of size `matrix_size`.
    """
    import numpy as np
    from qoc.standard import conjugate_transpose
    matrix = random_complex_matrix(matrix_size)
    return (matrix + conjugate_transpose(matrix)) / 2


if __name__ == "__main__":
    test_clip_control_norms()
    test_strip_slap()
    test_evolve_lindblad_discrete()
    test_grape_lindblad_discrete()
    test_get_lindbladian()
    test_interpolate_linear_points()
    test_magnus()
    test_rkdp5()
    test_evolve_schroedinger_discrete()
    test_grape_schroedinger_discrete()
