"""
mathutil.py - a module for math constants and methods
that the user should be exposed to

Most functions in this module use autograd.numpy so they are
autograd compatible.
"""

from functools import reduce

import autograd.numpy as anp
import numpy as np

### CONSTANTS ###

PAULI_X = anp.array([[0, 1], [1, 0]])
PAULI_Y = anp.array([[0, -1j], [1j, 0]])
PAULI_Z = anp.array([[1, 0], [0, -1]])


def get_creation_operator(d):
    """
    Construct the creation operator, truncated at level d.
    Args:
    d :: int - the level to truncate the operator at (d >= 1).
    Returns:
    creation_operator :: np.matrix - the creation operator at level d.
    """
    creation_operator = np.zeros((d, d))
    
    for i in range(1, d):
            creation_operator[i][i - 1] = np.sqrt(i)
            
    return creation_operator
      
def get_annihilation_operator(d):
    """
    Construct the annihilation operator, truncated at level d.
    Args:
    d :: int - the level to truncate the operator at (d >= 1).
    Returns:
    annihilation_operator :: np.matrix - the annihilation operator at level d.
    """
    annihilation_operator = np.zeros((d, d))
    
    for i in range(d - 1):
        annihilation_operator[i][i + 1] = np.sqrt(i + 1)

    return annihilation_operator


### SHORTHAND METHODS ###

commutator = lambda a, b: anp.matmul(a, b) - anp.matmul(b, a)
krons = lambda *matrices: reduce(anp.kron, matrices)
matmuls = lambda *matrices: reduce(anp.matmul, matrices)
conjugate_transpose = lambda matrix: anp.conjugate(anp.swapaxes(matrix, -1, -2))

### ISOMORPHISMS ###

# A row vecotr is np.array([[0, 1, 2]])
# A column vector is np.array([[0], [1], [2]])
column_vector_list_to_matrix = (lambda column_vector_list:
                                anp.hstack(column_vector_list))
matrix_to_column_vector_list = (lambda matrix:
                                anp.stack([anp.vstack(matrix[:, i])
                                           for i in range(matrix.shape[1])]))
# C -> R2
complex_to_real_imag_vec = lambda x: np.stack((np.real(x), np.imag(x)), axis=0)
real_imag_to_complex_vec = lambda x: x[0] + 1j * x[1]



_CA_TEST_COUNT = 1000

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: nothing
    """

    # Use the fact that (create)(annihilate) is the number operator.
    for i in range(1, _CA_TEST_COUNT):
        number_operator = np.zeros((_CA_TEST_COUNT, _CA_TEST_COUNT))
        for j in range(_CA_TEST_COUNT):
            number_operator[j][j] = j
        supposed_number_operator = np.matmul(get_creation_operator(i), get_annihilation_operator(i))
        assert number_operator.all() == supposed_number_operator.all()
        

if __name__ == "__main__":
    _tests()
