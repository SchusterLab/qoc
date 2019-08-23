"""
mathmethods.py - mathematical methods in physics
"""

### INTERPOLATION METHODS ###

def interpolate_linear(x1, x2, x3, y1, y2):
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

    Returns:
    y3 :: any - the interpolated value corresponding to x3, type
                is that resulting from composition of y1 and y2
    """
    return y1 + (((y2 - y1) / (x2 - x1)) * (x3 - x1))


### MODULE TESTS ###

_BIG = int(1e3)
_BIG_DIV_2 = _BIG / 2

def _test():
    """
    Run tests on the module.
    """
    import numpy as np
    
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
        itx3 = interpolate_linear(y1, y2, x1, x2, x3)
        assert(np.isclose(lx3, itx3))
    #ENDFOR


if __name__ == "__main__":
    _test()
