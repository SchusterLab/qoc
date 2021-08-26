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
def krylov(A,states,tol=2**-53):
    if tol==None:
        tol=2**-53
    if len(states.shape)<=2:
        states=states.flatten()
        box=expm_multiply(A,states,u_d=tol)
    else:
        states=states.reshape((states.shape[0]),states.shape[1])
        box=[]
        for i in range(states.shape[0]):
            box.append(expm_multiply(A,states[i],u_d=tol))
        box=np.array(box)
        box=box.reshape((states.shape[0]),states.shape[1],1)
    return box


def block_fre(A,E,state,tol):
    if tol==None:
        tol=2**-53
    HILBERT_SIZE = state.shape[0]
    if isspmatrix(A) is False:
        c = np.block([[A, E], [np.zeros_like(A), A]])
    else:
        c = bmat([[A, E], [None, A]]).tocsc()
    state = state.flatten()
    state0 = np.zeros_like(state)
    state = np.block([state0, state])

    state = expm_multiply(c, state,u_d=tol)
    state = state[0:HILBERT_SIZE]
    return state.reshape((HILBERT_SIZE, 1))
"""Compute the action of the matrix exponential.
"""

import numpy as np

import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.sputils import is_pydata_spmatrix

__all__ = ['expm_multiply']


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


def expm_multiply(A, B,u_d=2**-53):
    """
    Compute the action of the matrix exponential of A on B.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.

    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.

    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.

    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/
    """
    X = _expm_multiply_simple(A, B,u_d=u_d)
    return X


def _expm_multiply_simple(A, B, t=1.0, balance=False,u_d = 2 ** -53):
    """
    Compute the action of the matrix exponential at a single time point.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray
        :math:`e^{t A} B`

    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).

    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('the matrices A and B have incompatible shapes')
    ident = _ident_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    tol = u_d
    mu = _trace(A) / float(n)
    A = A - mu * ident
    A_1_norm = _exact_1_norm(A)
    if t * A_1_norm == 0:
        m_star, s_star = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
        m_star, s_star = _fragment_3_1(norm_info, n0, tol, ell=ell)
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s_star, tol, balance)


def _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None, balance=False):
    """
    A helper function.
    """
    if balance:
        raise NotImplementedError
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    F = B
    eta = np.exp(t*mu / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            B = coeff * A.dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            c=tol * _exact_inf_norm(F)
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F = eta * F
        B = F
    return F


# This table helps to compute bounds.
# They seem to have been difficult to calculate, involving symbolic
# manipulation of equations, followed by numerical root finding.
_theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }


def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Efficiently estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    #XXX Eventually turn this into an API function in the  _onenormest module,
    #XXX and remove its underscore,
    #XXX but wait until expm_multiply goes into scipy.
    return scipy.sparse.linalg.onenormest(aslinearoperator(A) ** p)

class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """
    def __init__(self, A, A_1_norm=None, ell=2, scale=1):
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        A_1_norm : float, optional
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        self._A = A
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}
        self._scale = scale

    def set_scale(self,scale):
        """
        Set the scale parameter.
        """
        self._scale = scale

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        if self._A_1_norm is None:
            self._A_1_norm = _exact_1_norm(self._A)
        return self._scale*self._A_1_norm

    def d(self, p):
        """
        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
        """
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, self._ell)
            self._d[p] = est ** (1.0 / p)
        return self._scale*self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p+1))

def _compute_cost_div_m(m, p, norm_info):
    """
    A helper function for computing bounds.

    This is equation (3.10).
    It measures cost in terms of the number of required matrix products.

    Parameters
    ----------
    m : int
        A valid key of _theta.
    p : int
        A matrix power.
    norm_info : LazyOperatorNormInfo
        Information about 1-norms of related operators.

    Returns
    -------
    cost_div_m : int
        Required number of matrix products divided by m.

    """
    return int(np.ceil(norm_info.alpha(p) / _theta[m]))


def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    best_m : int
        Related to bounds for error control.
    best_s : int
        Amount of scaling.

    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).

    """
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
        for m, theta in _theta.items():
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return best_m, best_s


def _condition_3_13(A_1_norm, n0, m_max, ell):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    A_1_norm : float
        The precomputed 1-norm of A.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    value : bool
        Indicates whether or not the condition has been met.

    Notes
    -----
    This is condition (3.13) in Al-Mohy and Higham (2011).

    """

    # This is the rhs of equation (3.12).
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)

    # Evaluate the condition (3.13).
    b = _theta[m_max] / float(n0 * m_max)
    return True

