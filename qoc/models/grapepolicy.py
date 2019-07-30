"""
grapepolicy.py - a module to define a class to encapsulate the choice
between grape methods
"""

from enum import Enum

class GrapeSchroedingerPolicy(Enum):
    """a class to encapsulate the choice between grape methods
    Such is computing that there is a trade-off between time and space.
    """

    TIME_EFFICIENT = 1
    MEMORY_EFFICIENT = 2

    def __str__(self):
        if self.value == 1:
            return "grape_policy_time"
        else:
            return "grape_policy_memory"


    def __repr__(self):
        return self.__str__()
