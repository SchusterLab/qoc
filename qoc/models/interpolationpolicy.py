"""
interpolationpolicy.py - a module to define a class to encapsulate
interpolation decisions for time discrete parameters
"""

from enum import Enum

class InterpolationPolicy(Enum):
    """
    a class to encapsulate interpolation decisions for time discrete parameters
    """
    LINEAR = 1

    def __repr__(self):
        return self.__str__()

    
    def __str__(self):
        if self.value == 1:
            return "interpolation_linear"

        

