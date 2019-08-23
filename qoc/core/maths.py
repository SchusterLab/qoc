"""
maths.py - a module for math methods
"""

import autograd.numpy as anp
import numpy as np

from qoc.models.interpolationpolicy import (InterpolationPolicy,)
from qoc.models.operationpolicy import (OperationPolicy,)
from qoc.standard import (commutator, conjugate_transpose,
                          matmuls,)

### INTERPOLATION METHODS ###

def interpolate_linear(y1, y2, x1, x2, x3):
    """
    Perform a linear interpolation of the point
    (x3, y3) using two points (x1, y1), (x2, y2).
    Args:
    y1 :: any - the independent variable dependent on x1
    y2 :: any - the independent variable dependent on x2, type
                must be composable with y1
    x1 :: float - the dependent variable on which y1 depends
    x2 :: float - the dependent variable on which y2 depends
    x3 :: float - the dependent variable on which y3 depends
    Returns:
    y3 :: any - the interpolated value corresponding to x3, type
                is that resulting from composition of y1 and y2
    """
    return y1 + (((y2 - y1) / (x2 - x1)) * (x3 - x1))


### MAGNUS EXPANSION METHODS ###

def magnus_m2(a1, dt):
    """
    a magnus expansion method of order two
    as seen in https://arxiv.org/abs/1709.06483
    Args:
    a1 :: numpy.ndarray - see paper
    dt :: float - see paper
    Returns:
    m2 :: numpy.ndarray - magnus expansion
    """
    return dt * a1


def magnus_m4(a1, a2, dt):
    """
    a magnus expansion method of order four
    as seen in https://arxiv.org/abs/1709.06483
    Args:
    a1 :: numpy.ndarray - see paper
    a2 :: numpy.ndarray - see paper
    dt :: float - see paper
    Returns:
    m4 :: numpy.ndarray - magnus expansion
    """
    return (anp.divide(dt, 2) * (a1 + a2) +
            anp.divide(anp.sqrt(3), 12) * anp.power(dt, 2) * commutator(a2, a1))
    

def magnus_m6(a1, a2, a3, dt):
    """
    a magnus expansion method of order six
    as seen in https://arxiv.org/abs/1709.06483
    Args:
    a1 :: numpy.ndarray - see paper
    a2 :: numpy.ndarray - see paper
    a3 :: numpy.ndarray - see paper
    dt :: float - see paper
    Returns:
    m6 :: numpy.ndarray - magnus expansion
    """
    b1 = dt * a2
    b2 = anp.divide(anp.sqrt(15), 3) * dt * (a3 - a1)
    b3 = anp.divide(10, 3) * dt * (a3 - 2 * a2 + a1)
    b1_b2_commutator = commutator(b1, b2)
    return (b1 + anp.divide(1, 2) * b3 + anp.divide(1, 240)
            * commutator(-20 * b1 - b3 + b1_b2_commutator,
                         b2 - anp.divide(1, 60)
                         * commutator(b1, 2 * b3 + b1_b2_commutator)))


### MAGNUS EXPANSION HAMILTONIAN EVALUATIONS ###

_MAGNUS_M4_C1 = anp.divide(1, 2) - anp.divide(anp.sqrt(3), 6)
_MAGNUS_M4_C2 = anp.divide(1, 2) + anp.divide(anp.sqrt(3), 6)
_MAGNUS_M6_C1 = anp.divide(1, 2) - anp.divide(anp.sqrt(15), 10)
_MAGNUS_M6_C2 = anp.divide(1, 2)
_MAGNUS_M6_C3 = anp.divide(1, 2) + anp.divide(anp.sqrt(15), 10)

def magnus_m2_linear(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Evaluate the m2 magnus expansion of the system hamiltonian.
    The linearly interpolated magnus expansion depends on
    the control parameters at time t.
    See https://arxiv.org/abs/1709.06483 for details.
    We take our own liberties here and do not evaluate at the midpoint
    between t and t + dt but instead evaluate at t.
    Args:
    hamiltonian :: (params :: numpy.ndarray, time :: float)
                    -> hamiltonian :: np.ndarray
        - the hamiltonian to expand
    params :: numpy.ndarray - time discrete parameters for the hamiltonian
        which should be those params from the total parameters array with
        indices specified by magnus_m2_linear_param_indices
    step :: int - an index into the params array about which to expand
                  the hamiltonian
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    sentinel :: bool - set to True if evaluating at the last step in params
    Returns:
    magnus :: numpy.ndarray - the m2 magnus expansion of the sytem hamiltonian
    """
    return magnus_m2(-1j * hamiltonian(params[0], t), dt)


def magnus_m2_linear_param_indices(hamiltonian, dt, params, step, t, sentinel=False):
    """
    The point of this paradigm is to figure out which params should be sent to
    the magnus expansion to be used for interpolation. That way, we only have to calculate
    the gradient of the magnus expansion with respect to the params used. However,
    we still keep the abstraction that any number of params may be used. In practice,
    we expect that only a few of the params near the step indexed into params will
    be used. Therefore, we expect to save memory and time.
    Args: see magnus_m2_linear
    Returns:
    indices :: numpy.ndarray - the indices of params that will be
        used in interpolation
    """
    return np.array([step])


def magnus_m4_linear(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Evaluate the m4 magnus expansion of the system hamiltonian.
    The linearly interpolated magnus expansion depends on the
    control parameters, between time t and t + dt
    which have params_left and params_right,
    respectively. See https://arxiv.org/abs/1709.06483 for details.
    Args:
    hamiltonian :: (params :: numpy.ndarray, time :: float)
                    -> hamiltonian :: np.ndarray
        - the hamiltonian to expand
    params :: numpy.ndarray - time discrete parameters for the hamiltonian
        which should be those params from the total parameters array with
        indices specified by magnus_m4_linear_param_indices
    step :: int - an index into the params array about which to expand
                  the hamiltonian
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    sentinel :: bool - set to True if evaluating at the last setp in params
    Returns:
    magnus :: numpy.ndarray - the m4 magnus expansion of the sytem hamiltonian
    """
    t1 = t + dt * _MAGNUS_M4_C1
    t2 = t + dt * _MAGNUS_M4_C2

    # Interpolate parameters.
    if sentinel:
        params_left = params[0] # step - 1
        params_right = params[1] # step
        params1 = interpolate_linear(params_left, params_right, t - dt, t, t1)
        params2 = interpolate_linear(params_left, params_right, t - dt, t, t2)
    else:
        params_left = params[0] # step
        params_right = params[1] # step + 1
        params1 = interpolate_linear(params_left, params_right, t, t + dt, t1)
        params2 = interpolate_linear(params_left, params_right, t, t + dt, t2)

    # Generate hamiltonians.
    a1 = -1j * hamiltonian(params1, t1)
    a2 = -1j * hamiltonian(params2, t2)
    
    return magnus_m4(a1, a2, dt)


def magnus_m4_linear_param_indices(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Args: see magnus_m4_linear
    Returns:
    indices :: numpy.ndarray - the indices of params that will be used
        in interpolation
    """
    if sentinel:
        return np.array([step - 1, step])
    else:
        return np.array([step, step + 1])


def magnus_m6_linear(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Evaluate the m6 magnus expansion of the system hamiltonian.
    The linearly interpolated magnus expansion depends on the
    control parameters, between time t and t + dt
    which have params_left and params_right,
    respectively. See https://arxiv.org/abs/1709.06483 for details.
    Args:
    hamiltonian :: (params :: numpy.ndarray, time :: float)
                    -> hamiltonian :: np.ndarray
        - the hamiltonian to expand
    params :: numpy.ndarray - time discrete parameters for the hamiltonian
        which should be those params from the total parameters array with
        indices specified by magnus_m6_linear_param_indices
    step :: int - an index into the params array about which to expand
                  the hamiltonian
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    sentinel :: bool - set to True if evaluating at the last step of params
    Returns:
    magnus :: numpy.ndarray - the m6 magnus expansion of the sytem hamiltonian
    """
    t1 = t + dt * _MAGNUS_M6_C1
    t2 = t + dt * _MAGNUS_M6_C2
    t3 = t + dt * _MAGNUS_M6_C3

    # Interpolate parameters.
    if sentinel:
        params_left = params[0] # step - 1
        params_right = params[1] # step
        params1 = interpolate_linear(params_left, params_right, t - dt, t, t1)
        params2 = interpolate_linear(params_left, params_right, t - dt, t, t2)
        params3 = interpolate_linear(params_left, params_right, t - dt, t, t3)
    else:
        params_left = params[0] # step
        params_right = params[1] # step + 1
        params1 = interpolate_linear(params_left, params_right, t, t + dt, t1)
        params2 = interpolate_linear(params_left, params_right, t, t + dt, t2)
        params3 = interpolate_linear(params_left, params_right, t, t + dt, t3)
    
    # Generate hamiltonians.
    a1 = -1j * hamiltonian(params1, t1)
    a2 = -1j * hamiltonian(params2, t2)
    a3 = -1j * hamiltonian(params3, t3)
    
    return magnus_m6(a1, a2, a3, dt)


def magnus_m6_linear_param_indices(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Args: see magnus_m6_linear
    Returns:
    indices :: numpy.ndarray - the indices of params that will be used
        in interpolation
    """
    if sentinel:
        return np.array([step - 1, step])
    else:
        return np.array([step, step + 1])


### LINDBLAD EVALUATIONS ###

def evolve_lindblad(controls, control_step, densities, dt,
                    hamiltonian, interpolation_policy, lindblad_operators,
                    time, control_sentinel=False,
                    operation_policy=OperationPolicy.CPU):
    """
    Use Runge-Kutta 4th order to evolve the density matrices to the next time step
    under the lindblad master equation.

    NOTATION:
     - t is time, c is controls, h is hamiltonian, g is dissipation constants,
       l is lindblad operators, k are the runge-kutta increments

    Args:
    controls :: ndarray - the controls that should be provided to the
        hamiltonian for the evolution    
    control_sentinel :: bool - set to True if this is the final control step,
        in which case control interpolation is performed on the last two
        control sets in the controls array
    control_step :: int - the index into the control array at which control
        interpolation should be performed
    densities :: ndarray - the probability density matrices to evolve
    dt :: float - the time step
    hamiltonian :: (controls :: ndarray, time :: float) -> hamiltonian :: ndarray
        - an autograd compatible function to generate the hamiltonian
          for the given controls and time
    interpolation_policy :: qoc.InterpolationPolicy - how parameters
        should be interpolated for intermediate time steps
    lindblad_operators :: (time :: float) -> (dissipartors :: ndarray, operators :: ndarray)
        - a function to generate the dissipation constants and lindblad operators
          for a given time
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.
    time :: float - the current evolution time

    Returns:
    densities :: ndarray - the densities evolved to `time + dt`
    """
    if control_sentinel:
        control_left = controls[control_step - 1]
        control_right = controls[control_step]
    else:
        control_left = controls[control_step]
        control_right = controls[control_step + 1]
    t1 = time
    t2 = time + 0.5 * dt
    t2 = t3
    t4 = time + dt
    c1 = control_left
    if interpolation_policy == InterpolationPolicy.LINEAR:
        c2 = interpolate_linear(control_left)
    else:
        raise ValueError("Unrecognized interpolation policy {}"
                         "".format(interpolation_policy))
    c3 = c2
    c4 = control_right
    h1 = hamiltonian(c1, t1)
    h2 = hamiltonian(c2, t2)
    h3 = h2
    h4 = hamiltonian(c4, t4)
    g1, l1 = lindblad_operators(t1)
    g2, l2 = lindblad_operators(t2)
    g3, l3 = lindblad_operators(t3)
    g4, l4 = lindblad_operators(t4)
    k1 = dt * get_lindbladian(densities, g1, h1, l1)
    k2 = dt * get_lindbladian(densities + 0.5 * k1, g2, h2, l2)
    k3 = dt * get_lindbladian(densities + 0.5 * k2, g3, h3, l3)
    k4 = dt * get_lindbladian(densities + k3, g4, h4, l4)

    densities = densities + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return densities


def get_lindbladian(densities, dissipators, hamiltonian, operators):
    """
    Compute the action of the lindblad operator on a single (set of)
    density matrix (matrices).

    Implemented by definition: https://en.wikipedia.org/wiki/Lindbladian

    Args:
    densities :: ndarray - the probability density matrices
    dissipators :: ndarray - the lindblad dissipators
    hamiltonian :: ndarray
    operators :: ndarray - the lindblad operators

    Returns:
    lindbladian :: ndarray - the lindbladian operator acting on the densities
    """
    operators_dagger = conjugate_transpose(operators)
    operators_product = matmuls(operators_dagger, operators)
    lindbladian = -1j * commutator(hamiltonian, densities)
    for i, operator in enumerate(operators):
        dissipator = dissipators[i]
        operator_dagger = operators_dagger[i]
        operator_product = operators_product[i]
        lindbladian = lindbladian + (dissipator * (matmuls(operator, densities, operator_dagger)
                                                   - 0.5 * matmuls(operator_product, densities)
                                                   - 0.5 * matmuls(densities, operator_product)))
    #ENDFOR
    return lindbladian


### MODULE TESTS ###


_BIG = int(1e3)
_BIG_DIV_2 = _BIG / 2

def _test():
    """
    Run tests on the module.
    """
    # Test the interpolation methods.
    for i in range(_BIG):
        # Generate a line with a constant slope between -5 and 5.
        line = lambda x: slope * x
        slope = anp.random.rand() * 10 - 5
        x1 = anp.random.rand() * _BIG - _BIG_DIV_2
        x2 = anp.random.rand() * _BIG - _BIG_DIV_2
        x3 = anp.random.rand() * _BIG - _BIG_DIV_2
        # Check that the trapezoid method approximates the line
        # exactly.
        y1 = line(x1)
        y2 = line(x2)
        lx3 = line(x3)
        itx3 = interpolate_linear(y1, y2, x1, x2, x3)
        assert(anp.isclose(lx3, itx3))

    # Test the magnus expansion methods.
    # These tests ensure the above methods were copied to code correclty.
    # They are hand checked. There may be a better way to test the methods.
    dt = 1.
    identity = anp.eye(2)
    assert(anp.allclose(magnus_m2(identity, dt), identity))
    assert(anp.allclose(magnus_m4(*([identity] * 2), dt), identity))
    assert(anp.allclose(magnus_m6(*([identity] * 3), dt), identity))
    dt = 2.
    a1 = anp.array([[2., 3.], [4., 5.]])
    a2 = anp.array([[9., 6.], [8., 7.]])
    a3 = anp.array([[12., 13.], [11., 10.]])
    assert(anp.allclose(magnus_m2(a1, dt),
                      anp.array([[4., 6.],
                                [8., 10.]])))
    assert(anp.allclose(magnus_m4(a1, a2, dt),
                      anp.array([[11., 22.85640646],
                                [-6.47520861, 12.]])))
    assert(anp.allclose(magnus_m6(a1, a2, a3, dt),
                      anp.array([[-241.71158615, 100.47657236],
                                [310.29160996, 263.71158615]])))

    # Test the magnus param indices methods.
    step_count = 20
    final_step = step_count - 1
    for step in range(step_count):
        sentinel = (step == final_step)
        assert(magnus_m2_linear_param_indices(None, None, None, step, None, sentinel=sentinel)
               == np.array((step,)))
        if sentinel:
            assert(np.allclose(magnus_m4_linear_param_indices(None, None, None, step,
                                                              None, sentinel=sentinel),
                               np.array((step - 1, step,))))
            assert(np.allclose(magnus_m6_linear_param_indices(None, None, None, step,
                                                  None, sentinel=sentinel),
                   np.array((step - 1, step,))))
        else:
            assert(np.allclose(magnus_m4_linear_param_indices(None, None, None, step,
                                                              None, sentinel=sentinel),
                               np.array((step, step + 1,))))
            assert(np.allclose(magnus_m6_linear_param_indices(None, None, None, step,
                                                              None, sentinel=sentinel),
                               np.array((step, step + 1,))))


if __name__ == "__main__":
    _test()
