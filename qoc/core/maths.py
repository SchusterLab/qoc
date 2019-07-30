"""
maths.py - a module for math methods
"""

import autograd.numpy as anp
import numpy as np

from qoc.util import commutator

# interpolation methods

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
    return y1 + anp.divide(y2 - y1, x2 - x1) * (x3 - x1)


# magnus expansion methods

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


# magnus expansion hamiltonian evaluations

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
    step :: int - an index into the params array about which to expand
                  the hamiltonian
    t :: float - the time at which the hamiltonian is being expanded
    dt :: float - the time step of the system evolution
    sentinel :: bool - set to True if evaluating at the last step in params
    Returns:
    magnus :: numpy.ndarray - the m2 magnus expansion of the sytem hamiltonian
    """
    return magnus_m2(hamiltonian(params[step], t), dt)


def magnus_m2_linear_param_indices(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Args: see magnus_m2_linear
    Returns:
    indices :: numpy.ndarray - the indices of params that were used in interpolation
    """
    return np.array(step)


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
        params_left = params[step - 1]
        params_right = params[step]
        params1 = interpolate_linear(params_left, params_right, t - dt, t, t1)
        params2 = interpolate_linear(params_left, params_right, t - dt, t, t2)
    else:
        params_left = params[step]
        params_right = params[step + 1]
        params1 = interpolate_linear(params_left, params_right, t, t + dt, t1)
        params2 = interpolate_linear(params_left, params_right, t, t + dt, t2)

    # Generate hamiltonians.
    a1 = hamiltonian(params1, t1)
    a2 = hamiltonian(params2, t2)
    
    return magnus_m4(a1, a2, dt)


def magnus_m4_linear_param_indices(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Args: see magnus_m4_linear
    Returns:
    indices :: numpy.ndarray - the indices of params that were used in interpolation
    """
    if sentinel:
        return np.array(step -1, step)
    else:
        return np.array(step, step + 1)


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
        params_left = params[step - 1]
        params_right = params[step]
        params1 = interpolate_linear(params_left, params_right, t - dt, t, t1)
        params2 = interpolate_linear(params_left, params_right, t - dt, t, t2)
        params2 = interpolate_linear(params_left, params_right, t - dt, t, t3)
    else:
        params_left = params[step]
        params_right = params[step + 1]
        params1 = interpolate_linear(params_left, params_right, t, t + dt, t1)
        params2 = interpolate_linear(params_left, params_right, t, t + dt, t2)
        params2 = interpolate_linear(params_left, params_right, t, t + dt, t3)
    
    # Generate hamiltonians.
    a1 = hamiltonian(params1, t1)
    a2 = hamiltonian(params2, t2)
    a3 = hamiltonian(params3, t3)
    
    return magnus_m6(a1, a2, a3, dt)


def magnus_m6_linear_param_indices(hamiltonian, dt, params, step, t, sentinel=False):
    """
    Args: see magnus_m6_linear
    Returns:
    indices :: numpy.ndarray - the indices of params that were used in interpolation
    """
    if sentinel:
        return np.array(step -1, step)
    else:
        return np.array(step, step + 1)


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


if __name__ == "__main__":
    _test()

