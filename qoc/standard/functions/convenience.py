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
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import bmat,isspmatrix

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
def krylov(A,states,if_AB=True):
    if if_AB==True:
        if len(states.shape)<=2:
            states=states.flatten()
            box=expm_multiply(A,states)
        else:
            states=states.reshape((states.shape[0]),states.shape[1])
            box=[]
            for i in range(states.shape[0]):
                box.append(expm_multiply(A,states[i]))
            box=np.array(box)
            box=box.reshape((states.shape[0]),states.shape[1],1)
        return box
    else:
        A=expm(A)
        state=np.dot(A.toarray(),states)
        return state.reshape(1,states.shape[1],1)

def block_fre(A,E,state,if_AB):
    HILBERT_SIZE = state.shape[0]
    if if_AB == True:

        if isspmatrix(A) is False:
            c = np.block([[A, E], [np.zeros_like(A), A]])
        else:
            c = bmat([[A, E], [None, A]]).tocsc()
        state = state.flatten()
        state0 = np.zeros_like(state)
        state = np.block([state0, state])
        if if_AB == True:
            state = expm_multiply(c, state)
        state = state[0:HILBERT_SIZE]
        return state.reshape((HILBERT_SIZE, 1))
    else:
        return np.dot(expm_frechet(A, E).todense(), state).reshape(HILBERT_SIZE,1)
def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):

    if method is None:
        method = 'SPS'
    if method == 'SPS':
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError('Unknown implementation %s' % method)
    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        return expm_frechet_AE


def expm_frechet_block_enlarge(A, E):
    """
    This is a helper function, mostly for testing and profiling.
    Return expm(A), frechet(A, E)
    """
    n = A.shape[0]
    M = np.vstack([
        np.hstack([A, E]),
        np.hstack([np.zeros_like(A), A])])
    expm_M = scipy.linalg.expm(M)
    return expm_M[:n, :n], expm_M[:n, n:]


"""
Maximal values ell_m of ||2**-s A|| such that the backward error bound
does not exceed 2**-53.
"""
ell_table_61 = (
        None,
        # 1
        2.11e-8,
        3.56e-4,
        1.08e-2,
        6.49e-2,
        2.00e-1,
        4.37e-1,
        7.83e-1,
        1.23e0,
        1.78e0,
        2.42e0,
        # 11
        3.13e0,
        3.90e0,
        4.74e0,
        5.63e0,
        6.56e0,
        7.52e0,
        8.53e0,
        9.56e0,
        1.06e1,
        1.17e1,
        )


# The b vectors and U and V are copypasted
# from scipy.sparse.linalg.matfuncs.py.
# M, Lu, Lv follow (6.11), (6.12), (6.13), (3.3)

def _diff_pade3(A, E, ident):
    b = (120., 60., 12., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    U = A.dot(b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
    Lv = b[2]*M2
    return U, V, Lu, Lv


def _diff_pade5(A, E, ident):
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[5]*M4 + b[3]*M2) +
            E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade7(A, E, ident):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade9(A, E, ident):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    A8 = np.dot(A4, A4)
    M8 = np.dot(A4, M4) + np.dot(M4, A4)
    U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def expm_frechet_algo_64(A,E ):
    n = A.shape[0]
    s = None
    ident = np.identity(n)
    A_norm_1 = scipy.sparse.linalg.norm(A, 1)
    m_pade_pairs = (
            (3, _diff_pade3),
            (5, _diff_pade5),
            (7, _diff_pade7),
            (9, _diff_pade9))
    for m, pade in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break
    if s is None:
        # scaling
        s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
        del A_norm_1
        # pade order 13
        A2 = np.dot(A * 2.0 ** -s ,A * 2.0 ** -s)
        M2 = np.dot(A * 2.0 ** -s, E * 2.0 ** -s) + np.dot(E * 2.0 ** -s, A * 2.0 ** -s)
        A4 = np.dot(A2, A2)
        M4 = np.dot(A2, M2) + np.dot(M2, A2)
        A6 = np.dot(A2, A4)
        M6 = np.dot(A4, M2) + np.dot(M4, A2)
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
             1187353796428800., 129060195264000., 10559470521600.,
             670442572800., 33522128640., 1323241920., 40840800., 960960.,
             16380., 182., 1.)
        W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
        W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        del ident
        A2=b[8] * A2
        A2=A2+b[10] * A4
        del A4
        Z1=b[12] * A6+A2
        del A2#10
        W = np.dot(A6, W1) + W2
        del W2
        V = np.dot(A6, Z1) + Z2
        del Z2
        Lw1 =  b[9] * M2
        Lw1=Lw1+b[11] * M4
        Lw1=Lw1+b[13] * M6
        Lw1=np.dot(A6, Lw1)+ np.dot(M6, W1)
        del W1
        M6=b[7] * M6
        M4=b[5] * M4
        M2=b[3] * M2

        Lw = Lw1+M2+M4+M6
        del Lw1
        M6=M6/b[7]
        M4=M4/b[5]
        M2=M2/b[3]
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        M2=b[2] * M2
        M2=b[4] * M4+M2
        M2 = b[6] * M6 +M2
        Lz2 = M2#13
        del M2,M4

        Lu = np.dot(A * 2.0 ** -s, Lw) +np.dot(E * 2.0 ** -s, W)
        del Lw
        U = np.dot(A * 2.0 ** -s, W)
        del W  # 11
        Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
        del A6,M6,Z1,Lz2,Lz1

    # factor once and solve twice
    if isspmatrix(U):
        lu_piv = sla.splu(-U + V)
        R = sla.spsolve(lu_piv, U + V)
        L = sla.spsolve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
    lu_piv = scipy.linalg.lu_factor(-U + V)
    R = scipy.linalg.lu_solve(lu_piv, U + V)
    L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
    del lu_piv, Lu, Lv, U, V
    # squaring
    for k in range(s):
        L = np.dot(R, L) + np.dot(L, R)
        R = np.dot(R, R)
    return  R,L