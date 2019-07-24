"""
mathutil.py - a module for math constants and methods
that the user should be exposed to
"""

from functools import reduce

import numpy as np

# constants

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])

# shorthand methods

commutator = lambda a, b: np.matmul(a, b) - np.matmul(b, a)
krons = lambda *matrices: reduce(np.kron, matrices)
matmuls = lambda *matrices: reduce(np.matmul, matrices)

