"""
programtype.py - This module defines a class to specify the active program.
"""

from enum import Enum

class ProgramType(Enum):
    """
    This class specifies the active program.
    """
    EVOLVE = 1
    GRAPE = 2


    def __repr__(self):
        return self.__str__()


    def __str__(self):
        if self.value == 1:
            string = "evolve"
        else:
            string = "grape"

        return string
