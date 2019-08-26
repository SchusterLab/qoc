"""
performancepolicy.py - This module defines a class to encapsulate the choice
between performance options.
"""

from enum import Enum

class PerformancePolicy(Enum):
    """
    This class encapsulates the choice between performance options:
    such is computing that there is a trade-off between time and space.
    """

    TIME = 1
    MEMORY = 2

    def __str__(self):
        if self.value == 1:
            return "performance_policy_time"
        else:
            return "performance_policy_memory"


    def __repr__(self):
        return self.__str__()
