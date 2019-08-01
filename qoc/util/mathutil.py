"""
mathutil.py - a module for math constants and methods
that the user should be exposed to

Most functions in this module use autograd.numpy so they are
autograd compatible.
"""

from functools import reduce

import autograd.numpy as anp
import numpy as np

# constants

PAULI_X = anp.array([[0, 1], [1, 0]])
PAULI_Y = anp.array([[0, -1j], [1j, 0]])
PAULI_Z = anp.array([[1, 0], [0, -1]])

# shorthand methods

commutator = lambda a, b: anp.matmul(a, b) - anp.matmul(b, a)
krons = lambda *matrices: reduce(anp.kron, matrices)
matmuls = lambda *matrices: reduce(anp.matmul, matrices)
conjugate_transpose = lambda matrix: anp.conjugate(anp.swapaxes(matrix, -1, -2))

# isomorphisms

# A column vector is NOT a row vector anp.array([0, 1, 2])
# A column vector is specified as anp.array([[0], [1], [2]])
column_vector_list_to_matrix = (lambda column_vector_list:
                                anp.hstack(column_vector_list))
matrix_to_column_vector_list = (lambda matrix:
                                anp.stack([anp.vstack(matrix[:, i])
                                           for i in range(matrix.shape[1])]))
# C -> R^2
complex_to_real_imag_vec = lambda x: np.stack((np.real(x), np.imag(x)), axis=0)
real_imag_to_complex_vec = lambda x: x[0] + 1j * x[1]
