"""
convenience.py - definitions of common computations
All functions in this module that are exported, 
i.e. those that don't begin with '_', are autograd compatible.
"""
from scipy.sparse import isspmatrix, linalg as sla
import scipy as scipy
from functools import reduce
from scipy.sparse.linalg import expm
from autograd.extend import defvjp, primitive
import autograd.numpy as anp
import numpy as np
import scipy.linalg as la
from scipy.sparse import bmat,isspmatrix,identity

from pyexpokit import expmv


### COMPUTATIONS ###

def commutator(a, b):
    """
    Compute the commutator of two matrices.

    Arguments:
    a :: numpy.ndarray - the left matrix
    b :: numpy.ndarray - the right matrix

    Returns:
    _commutator :: numpy.ndarray - the commutator of a and b
    """
    commutator_ = anp.matmul(a, b) - anp.matmul(b, a)

    return commutator_


def conjugate_transpose(matrix):
    """
    Compute the conjugate transpose of a matrix.
    Args:
    matrix :: numpy.ndarray - the matrix to compute
        the conjugate transpose of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _conjugate_tranpose :: numpy.ndarray the conjugate transpose
        of matrix
    """
    conjugate_transpose_ = anp.conjugate(anp.swapaxes(matrix, -1, -2))
    
    return conjugate_transpose_


def krons(*matrices):
    """
    Compute the kronecker product of a list of matrices.
    Args:
    matrices :: numpy.ndarray - the list of matrices to
        compute the kronecker product of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    """
    krons_ = reduce(anp.kron, matrices)

    return krons_


def matmuls(*matrices):
    """
    Compute the kronecker product of a list of matrices.
    Args:
    matrices :: numpy.ndarray - the list of matrices to
        compute the kronecker product of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    """
    matmuls_ = reduce(anp.matmul, matrices)

    return matmuls_


def rms_norm(array):
    """
    Compute the rms norm of the array.

    Arguments:
    array :: ndarray (N) - The array to compute the norm of.

    Returns:
    norm :: float - The rms norm of the array.
    """
    square_norm = anp.sum(array * anp.conjugate(array))
    size = anp.prod(anp.shape(array))
    rms_norm_ = anp.sqrt(square_norm / size)
    
    return rms_norm_


### ISOMORPHISMS ###

# A row vector is np.array([[0, 1, 2]])
# A column vector is np.array([[0], [1], [2]])
column_vector_list_to_matrix = (lambda column_vector_list:
                                anp.hstack(column_vector_list))


matrix_to_column_vector_list = (lambda matrix:
                                anp.stack([anp.vstack(matrix[:, i])
                                           for i in range(matrix.shape[1])]))
def krylov(dt, A,states,tol=2**-53):
    if tol==None:
        tol=2**-53
    if len(states.shape)<=2:
        states=states.flatten()
        box= expmv(dt, A, states, tol)
    else:
        states=states.reshape((states.shape[0]),states.shape[1])
        box=[]
        for i in range(states.shape[0]):
            box.append(expmv(dt, A, states[i], tol))
        box=np.array(box)
        box=box.reshape((states.shape[0]),states.shape[1],1)
    return box


def block_fre(dt, A, E, state, tol):
    if tol is None:
        tol = 2**-53
    HILBERT_SIZE = state.shape[0]
    if isspmatrix(A) is False:
        c = np.block([[A, E], [np.zeros_like(A), A]])
    else:
        c = bmat([[A, E], [None, A]]).tocsc()
    state = state.flatten()
    state0 = np.zeros_like(state)
    state = np.block([state0, state])

    state = expmv(dt, c, state, tol)
    state = state[0:HILBERT_SIZE]
    return state.reshape((HILBERT_SIZE, 1))

"""
Compute the action of the matrix exponential.
"""
import numpy as np

import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.sputils import is_pydata_spmatrix
def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=1))
    else:
        return np.linalg.norm(A, np.inf)


def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=0).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=0))
    else:
        return np.linalg.norm(A, 1)


def _trace(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return A.diagonal().sum()
    elif is_pydata_spmatrix(A):
        return A.to_scipy_sparse().diagonal().sum()
    else:
        return np.trace(A)


def _ident_like(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.isspmatrix(A):
        return scipy.sparse.construct.eye(A.shape[0], A.shape[1],
                dtype=A.dtype, format=A.format)
    elif is_pydata_spmatrix(A):
        import sparse
        return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)


def expm_multiply(A, B,u_d=1e-8):
    X = _expm_multiply_simple(A, B,tol=u_d)
    return X


def _expm_multiply_simple(A, B,tol= 1e-8):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('the matrices A and B have incompatible shapes')
    ident = _ident_like(A)
    n = A.shape[0]
    mu = _trace(A) / float(n)
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    if  A_1_norm == 0:
        m_star, s_star = 0, 1
    else:
        m_star,s_star=determ_s(A_1_norm,tol)
    return _expm_multiply_simple_core(A, B, mu, m_star, s_star, tol)


def _expm_multiply_simple_core(A, B,mu, m_star, s, tol=None,):
    """
    A helper function.
    """
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = B
    eta = np.exp(mu / float(s))
    for i in range(s):
        for j in range(m_star):
            coeff = 1 / float(s*(j+1))
            B = coeff * A.dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            if c2 <= tol * _exact_inf_norm(F):
                break
        F = eta * F
        B = F
    return F
theta_m=[
31.54144013,
29.35699609,
26.61684407,
24.44814683,
22.57093824,
20.70558923,
17.45584544,
14.23680405,
12.44158958,
9.86414502,
7.62996736,
4.75069937,
2.21018887,
]
max_m=[
    115,
    109,
    101,
    95,
    90,
    85,
    75,
    65,
    60,
    52,
    45,
    35,
    25
]
tol_table=[
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
1e-9,
1e-10,
1e-11,
1e-12,
1e-13,
1e-14,
1e-15,
1e-16,
]
def determ_s(norm,tol):
    for i in range(len(tol_table)):
        if tol>=tol_table[i]:
            m_max=max_m[i]
            theta=theta_m[i]
            break
    s = int(np.ceil(norm/ theta))
    return m_max,s
