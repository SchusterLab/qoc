"""
magnusorder.py - a module to define a class to encapsulate the choice
of the magnus expansion order
"""

from enum import Enum

class MagnusOrder(Enum):
    """a class to encapsulate the choice of the magnus expansion order
    """
    TWO = 1
    FOUR = 2
    SIX = 3

    def __str__(self):
        if self.value == 1:
            return "magnus_order_two"
        elif self.value == 2:
            return "magnus_order_four"
        else:
            return "magnus_order_six"
