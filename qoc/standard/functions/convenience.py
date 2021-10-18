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

from quspin.tools.lanczos import lanczos_iter, expm_lanczos


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


# Krylov subspace dimension
krylov_dim = 10
def krylov(A,states,tol=2**-53):
    if tol==None:
        tol=2**-53
    if len(states.shape)<=2:
        states=states.flatten()
        E, V, Q_T = lanczos_iter(A, states, krylov_dim)
        box=expm_lanczos(E, V, Q_T, a=1.0)
    else:
        states=states.reshape((states.shape[0]),states.shape[1])
        box=[]
        for i in range(states.shape[0]):
            E, V, Q_T = lanczos_iter(A, states[i], krylov_dim)
            box.append(expm_lanczos(E, V, Q_T, a=1.0))
        box=np.array(box)
        box=box.reshape((states.shape[0]),states.shape[1],1)
    return box


def block_fre(A,E,state,tol):
    if tol==None:
        tol=2**-53
    a=state.shape
    HILBERT_SIZE = state.shape[0]
    if isspmatrix(A) is False:
        c = np.block([[A, E], [np.zeros_like(A), A]])
    else:
        c = bmat([[A, E], [None, A]]).tocsc()
    state = state.flatten()
    state0 = np.zeros_like(state)
    state = np.block([state0, state])

    Eig, V, Q_T = lanczos_iter(c, state, krylov_dim)
    state = expm_lanczos(Eig, V, Q_T, a=1.0)
    new = state[HILBERT_SIZE:2*HILBERT_SIZE]
    state = state[0:HILBERT_SIZE]

    return state.reshape((HILBERT_SIZE, 1)),new.reshape((HILBERT_SIZE, 1))

"""Compute the action of the matrix exponential.
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

def get_s(A,b,tol):
    s=1
    if A.dtype==np.complex256:
        s=np.ceil(_exact_inf_norm(A))
    else:
        while(1):
            a=np.log10(tol)
            tol_power = np.ceil(np.log10(tol))
            norm_A = _exact_inf_norm(A)/s
            norm_b= _exact_inf_norm(b)
            max_term_notation=np.floor(norm_A)
            max_term=1
            for i in range(1,np.int(max_term_notation)):
                max_term=max_term*norm_A*norm_b/i
                max_power = np.ceil(np.log10(max_term))
                if max_power>30:
                    break
            max_power = np.ceil(np.log10(max_term))
            if max_power-16<=tol_power:
                break
            s=s+1
    return s
def expm_multiply(A, B, u_d=None):
    """
    A helper function.
    """
    tol=u_d
    ident = _ident_like(A)
    n = A.shape[0]
    mu = _trace(A) / float(n)
    A = A - mu * ident
    if tol is None:
        tol =1e-16
    s=get_s(A,B,tol)
    F = B
    for i in range(int(s)):
        j=0
        eta = np.exp(mu / float(s))
        while(1):
            coeff = s*(j+1)
            B =  A.dot(B)/coeff
            c2 = _exact_inf_norm(B)
            F = F + B
            total_norm=_exact_inf_norm(F)
            if c2/total_norm<tol:
                break
            j=j+1
        F = eta * F
        B = F
    return F