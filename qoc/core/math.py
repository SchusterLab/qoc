"""
math.py - a module for math methods
"""

import numpy as np

def interpolate_trapezoid(y1, y2, x1, x2, x3):
    """
    Perform a trapezoidal interpolation of the point
    (x3, y3) using two points (x1, y1), (x2, y2).
    Args:
    y1 :: any - the independent variable dependent on x1
    y2 :: any - the independent variable dependent on x2, type
                must be subtractable with y1
    x1 :: float - the dependent variable on which y1 depends
    x2 :: float - the dependent variable on which y2 depends
    x3 :: float - the dependent variable on which y3 depends
    Returns:
    y3 :: any - the interpolated value corresponding to x3, type
                is that resulting from subtraction of y1 and y2
    """
    return y1 + np.divide(y1 - y2, x1 - x2) * (x1 - x3)


def magnus_order_two():
    pass


def _test():
    """
    Run tests on the module.
    Args: none
    Returns: none
    """
    pass

if __name__ == "__main__":
    _test()

    
