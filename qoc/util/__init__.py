"""
util - a directory for methods that may be used in the
main module but should also be exposed to the user
"""

from qoc.util.mathutil import (commutator, krons, matmuls)

__all__ = ["commutator", "krons", "matmuls"]
