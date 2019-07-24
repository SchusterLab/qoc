"""
magnusmethod.py - a module to define a class to encapsulate the choice
of the magnus expansion method
"""

from enum import Enum

class MagnusMethod(Enum):
    """a class to encapsulate the choice of the magnus expansion method,
    see https://arxiv.org/abs/1709.06483
    """
    M2 = 1
    M4 = 2
    M6 = 3

    def __str__(self):
        if self.value == 1:
            return "magnus_order_two"
        elif self.value == 2:
            return "magnus_order_four"
        else:
            return "magnus_order_six"
