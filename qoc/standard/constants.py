"""
constants.py - This module defines constants.
"""

import numpy as np

### CONSTANTS ###

SIGMA_X = np.array(((0, 1), (1, 0)))
SIGMA_Y = np.array(((0, -1j), (1j, 0)))
SIGMA_Z = np.array(((1, 0), (0, -1)))
SIGMA_PLUS = np.array(((0, 2), (0, 0))) # SIGMA_X + i * SIGMA_Y
SIGMA_MINUS = np.array(((0, 0), (2, 0))) # SIGMA_X - i * SIGMA_Y


### GENERATIVE CONSTANTS ###

def get_creation_operator(size):
    """
    Construct the creation operator with the given matrix size.

    Arguments:
    size :: int - This is the size to truncate the operator at. This
        value should be g.t.e. 1.

    Returns:
    creation_operator :: ndarray (size, size)
        - The creation operator at level `size`.
    """
    return np.diag(np.sqrt(np.arange(1, size)), k=-1)


def get_annihilation_operator(size):
    """
    Construct the annihilation operator with the given matrix size.

    Arguments:
    size :: int - This is hte size to truncate the operator at. This value
        should be g.t.e. 1.

    Returns:
    annihilation_operator :: ndarray (size, size)
        - The annihilation operator at level `size`.
    """
    return np.diag(np.sqrt(np.arange(1, size)), k=1)


def get_eij(i, j, size):
    """
    Construct the square matrix of `size`
    where all entries are zero
    except for the element at row i column j which is one.

    Arguments:
    i :: int - the row of the unit element
    j :: int - the column of the unit element
    size :: int - the size of the matrix

    Returns:
    eij :: ndarray (size, size)
        - The requested Eij matrix.
    """
    eij = np.zeros((size, size))
    eij[i, j] = 1
    return eij


### MODULE TESTS ###

_BIG = 100

def _tests():
    """
    Run tests on the module.
    """
    # Use the fact that (create)(annihilate) is the number operator.
    for i in range(1, _BIG):
        analytic_number_operator = np.diag(np.arange(i))
        generated_number_operator = np.matmul(get_creation_operator(i), get_annihilation_operator(i))
        assert np.allclose(generated_number_operator, analytic_number_operator)
        

if __name__ == "__main__":
    _tests()
