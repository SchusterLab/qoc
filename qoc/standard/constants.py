"""
constants.py - This module defines constants.
"""

import numpy as np
from scipy.sparse import dia_matrix
from qutip import coherent
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

def harmonic(H_size):
    diagnol = np.arange(H_size)
    up_diagnol = np.sqrt(diagnol)
    low_diagnol = np.sqrt(np.arange(1, H_size + 1))
    a= dia_matrix(([ up_diagnol], [ 1]), shape=(H_size, H_size)).tocsc()
    a_dag=dia_matrix(([ low_diagnol], [ -1]), shape=(H_size, H_size)).tocsc()
    return a_dag,a

def transmon(w_01,anharmonicity,H_size):
    from scipy.sparse.linalg import expm_multiply
    b_dag,b=harmonic(H_size=H_size)
    H0=b_dag.dot(b)
    diagnol=np.ones(H_size)
    I= dia_matrix(([ diagnol], [ 0]), shape=(H_size, H_size)).tocsc()
    H0=w_01*H0+anharmonicity*H0*(H0-I)
    state=np.ones(H_size)
    expm_multiply(H0,state)
    return H0,b_dag,b

def coherent_state(N,alpha):
    state=np.array(coherent(N,alpha))
    return state.reshape((1,state.shape[0],1))
