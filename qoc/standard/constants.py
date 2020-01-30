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
    creation_operator :: ndarray (size x size)
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
    annihilation_operator :: ndarray (size x size)
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
    eij :: ndarray (size x size)
        - The requested Eij matrix.
    """
    eij = np.zeros((size, size))
    eij[i, j] = 1
    return eij


def RX(theta):
    """
    The X rotation gate.
    
    Arguments:
    theta :: float - the argument of the gate
    
    Returns:
    gate :: ndarray(2 x 2) - the X rotation gate with
        the given argument
    """
    return np.array(((np.cos(theta / 2), -1j * np.sin(theta / 2)),
                     (-1j * np.sin(theta / 2), np.cos(theta / 2))))


def RY(theta):
    """
    The Y rotation gate.
    
    Arguments:
    theta :: float - the argument of the gate
    
    Returns:
    gate :: ndarray(2 x 2) - the Y rotation gate with
        the given argument
    """
    return np.array(((np.cos(theta / 2), -np.sin(theta / 2)),
                     (np.sin(theta / 2), np.cos(theta / 2))))


def RZ(theta):
    """
    The Z rotation gate.
    
    Arguments:
    theta :: float - the argument of the gate
    
    Returns:
    gate :: ndarray(2 x 2) - the Z rotation gate with
        the given argument
    """
    return np.array(((np.cos(theta / 2) - 1j * np.sin(theta / 2), 0),
                     (0, np.cos(theta / 2) + 1j * np.sin(theta / 2))))
