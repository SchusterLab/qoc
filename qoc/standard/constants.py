"""
constants.py - definitions of common constants
"""

import numpy as np

### CONSTANTS ###

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])


### GENERATIVE CONSTANTS ###

def get_creation_operator(size):
    """
    Construct the creation operator with the given matrix size.
    Args:
    size :: int - the matrix size to truncate the operator at (size >= 1).
    Returns:
    creation_operator :: np.ndarray - the creation operator at level size.
    """
    creation_operator = np.zeros((size, size))
    
    for i in range(1, size):
            creation_operator[i, i - 1] = np.sqrt(i)
            
    return creation_operator


def get_annihilation_operator(size):
    """
    Construct the annihilation operator with the given matrix size.
    Args:
    size :: int - the matrix size to truncate the operator at (size >= 1).
    Returns:
    annihilation_operator :: np.ndarray - the annihilation operator at level size.
    """
    annihilation_operator = np.zeros((size, size))
    
    for i in range(size - 1):
        annihilation_operator[i, i + 1] = np.sqrt(i + 1)

    return annihilation_operator


def get_eij(i, j, size):
    """
    Construct the square matrix of the given size
    where all entries are zero
    except for the element at row i column j which is one.
    Args:
    i :: int - the row of the unit element
    j :: int - the column of the unit element
    size :: int - the size of the matrix
    Returns:
    Eij :: np.ndarray - the requested Eij matrix
    """
    eij = np.zeros((size, size))
    eij[i, j] = 1
    return eij


### MODULE TESTS ###

_BIG = 100

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: none
    """

    # Use the fact that (create)(annihilate) is the number operator.
    for i in range(1, _BIG):
        number_operator = np.zeros((i, i))
        for j in range(i):
            number_operator[j][j] = j
        supposed_number_operator = np.matmul(get_creation_operator(i), get_annihilation_operator(i))
        assert np.allclose(number_operator, supposed_number_operator)
        

if __name__ == "__main__":
    _tests()
