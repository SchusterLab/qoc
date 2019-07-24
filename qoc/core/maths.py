"""
maths.py - a module for math methods
"""

import numpy as np

from qoc.util import commutator

# interpolation methods

def interpolate_trapezoid(y1, y2, x1, x2, x3):
    """
    Perform a trapezoidal interpolation of the point
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
    return y1 + np.divide(y2 - y1, x2 - x1) * (x3 - x1)


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
    return (np.divide(dt, 2) * (a1 + a2) +
            np.divide(np.sqrt(3), 12) * np.power(dt, 2) * commutator(a2, a1))
    

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
    b2 = np.divide(np.sqrt(15), 3) * dt * (a3 - a1)
    b3 = np.divide(10, 3) * dt * (a3 - 2 * a2 + a1)
    b1_b2_commutator = commutator(b1, b2)
    return (b1 + np.divide(1, 2) * b3 + np.divide(1, 240)
            * commutator(-20 * b1 - b3 + b1_b2_commutator,
                         b2 - np.divide(1, 60)
                         * commutator(b1, 2 * b3 + b1_b2_commutator)))


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
        itx3 = interpolate_trapezoid(y1, y2, x1, x2, x3)
        assert(np.isclose(lx3, itx3))

    # Test the magnus expansion methods.
    # These tests ensure the above methods were copied to code correclty.
    # They are hand checked. There may be a better way to test the methods.
    dt = 1.
    identity = np.eye(2)
    assert(np.allclose(magnus_m2(identity, dt), identity))
    assert(np.allclose(magnus_m4(*([identity] * 2), dt), identity))
    assert(np.allclose(magnus_m6(*([identity] * 3), dt), identity))
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


if __name__ == "__main__":
    _test()

    
