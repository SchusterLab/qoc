"""
core - a directory for the primary functionality exposed by qoc
"""

from .lindbladdiscrete import (evolve_lindblad_discrete,
                               grape_lindblad_discrete,)
from .schroedingerdiscrete import (evolve_schroedinger_discrete,
                                   grape_schroedinger_discrete,)
from .common import expm_frechet

__all__ = [
    "evolve_lindblad_discrete",
    "grape_lindblad_discrete",
    "evolve_schroedinger_discrete",
    "grape_schroedinger_discrete","expm_frechet"

]
