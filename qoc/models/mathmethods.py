"""
mathmethods.py - mathematical methods in physics
"""

import numpy as np

from qoc.models.operationpolicy import OperationPolicy
from qoc.standard.functions.convenience import (commutator, conjugate_transpose,
                                                matmuls,)

### INTERPOLATION METHODS ###

def interpolate_linear(x1, x2, x3, y1, y2,
                       operation_policy=OperationPolicy.CPU):
    """
    Perform a linear interpolation of the point
    (x3, y3) using two points (x1, y1), (x2, y2).

    Args:
    x1 :: float - the dependent variable on which y1 depends
    x2 :: float - the dependent variable on which y2 depends
    x3 :: float - the dependent variable on which y3 depends

    y1 :: any - the independent variable dependent on x1
    y2 :: any - the independent variable dependent on x2, type
                must be composable with y1
    operation_policy

    Returns:
    y3 :: any - the interpolated value corresponding to x3, type
                is that resulting from composition of y1 and y2
    """
    return y1 + (((y2 - y1) / (x2 - x1)) * (x3 - x1))


### MAGNUS EXPANSION METHODS ###

def magnus_m2(a1, dt, operation_policy=OperationPolicy.CPU):
    """
    a magnus expansion method of order two
    as seen in https://arxiv.org/abs/1709.06483

    Args:
    a1 :: numpy.ndarray - see paper
    dt :: float - see paper
    operation_policy

    Returns:
    m2 :: numpy.ndarray - magnus expansion
    """
    return dt * a1


_M4_C0 = np.divide(np.sqrt(3), 12)
def magnus_m4(a1, a2, dt, operation_policy=OperationPolicy.CPU):
    """
    a magnus expansion method of order four
    as seen in https://arxiv.org/abs/1709.06483
    Args:
    a1 :: numpy.ndarray - see paper
    a2 :: numpy.ndarray - see paper
    dt :: float - see paper
    operation_policy

    Returns:
    m4 :: numpy.ndarray - magnus expansion
    """
    return ((dt / 2) * (a1 + a2) +
            _M4_C0 * np.power(dt, 2) * commutator(a2, a1,
                                                  operation_policy=operation_policy))
    

_M6_C0 = np.divide(np.sqrt(15), 3)
_M6_C1 = np.divide(10, 3)
_M6_C2 = np.divide(1, 2)
_M6_C3 = np.divide(1, 240)
_M6_C4 = np.divide(1, 60)
def magnus_m6(a1, a2, a3, dt, operation_policy=OperationPolicy.CPU):
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
    b2 = _M6_C0 * dt * (a3 - a1)
    b3 = _M6_C1 * dt * (a3 - 2 * a2 + a1)
    b1_b2_commutator = commutator(b1, b2, operation_policy=operation_policy)
    return (b1 + _M6_C2 * b3 + _M6_C3
            * commutator(-20 * b1 - b3 + b1_b2_commutator,
                         b2 - _M6_C4
                         * commutator(b1, 2 * b3 + b1_b2_commutator,
                                      operation_policy=operation_policy),
                         operation_policy=operation_policy))


### LINDBLAD METHODS ###

def get_lindbladian(densities, dissipators=None, hamiltonian=None,
                    operators=None,
                    operation_policy=OperationPolicy.CPU):
    """
    Compute the action of the lindblad equation on a single (set of)
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
        slope = np.random.rand() * 10 - 5
        x1 = np.random.rand() * _BIG - _BIG_DIV_2
        x2 = np.random.rand() * _BIG - _BIG_DIV_2
        x3 = np.random.rand() * _BIG - _BIG_DIV_2
        # Check that the trapezoid method approximates the line
        # exactly.
        y1 = line(x1)
        y2 = line(x2)
        lx3 = line(x3)
        itx3 = interpolate_linear(x1, x2, x3, y1, y2)
        assert(np.isclose(lx3, itx3))
    #ENDFOR

    # Test the magnus expansion methods.
    # These tests ensure the above methods were copied to code correclty.
    # They are hand checked. There may be a better way to test the methods.
    dt = 1.
    identity = np.eye(2)
    assert(np.allclose(magnus_m2(identity, dt), identity))
    assert(np.allclose(magnus_m4(identity, identity, dt), identity))
    assert(np.allclose(magnus_m6(identity, identity, identity, dt), identity))
    dt = 2.
    a1 = np.array([[2., 3.], [4., 5.]])
    a2 = np.array([[9., 6.], [8., 7.]])
    a3 = np.array([[12., 13.], [11., 10.]])
    assert(np.allclose(magnus_m2(a1, dt),
                      np.array([[4., 6.],
                                [8., 10.]])))
    assert(np.allclose(magnus_m4(a1, a2, dt),
                      np.array([[11., 22.85640646],
                                [-6.47520861, 12.]])))
    assert(np.allclose(magnus_m6(a1, a2, a3, dt),
                      np.array([[-241.71158615, 100.47657236],
                                [310.29160996, 263.71158615]])))

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

