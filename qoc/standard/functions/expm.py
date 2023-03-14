"""
expm.py - a module for all things e^M
"""

from autograd.extend import (defvjp as autograd_defvjp,
                             primitive as autograd_primitive)
import autograd.numpy as anp
import numpy as np
import scipy.linalg as la


### EXPM IMPLEMENTATION DUE TO HIGHAM 2005 ###

# Pade approximants from algorithm 2.3.
B = (
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
)

def one_norm(a):
    """
    Return the one-norm of the matrix.

    References:
    [0] https://www.mathworks.com/help/dsp/ref/matrix1norm.html

    Arguments:
    a :: ndarray(N x N) - The matrix to compute the one norm of.
    
    Returns:
    one_norm_a :: float - The one norm of a.
    """
    return anp.max(anp.sum(anp.abs(a), axis=0))
    

def pade3(a, i):
    a2 = anp.matmul(a, a)
    u = anp.matmul(a, B[2] * a2) + B[1] * a
    v = B[2] * a2 + B[0] * i
    return u, v


def pade5(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    u = anp.matmul(a, B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade7(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    a6 = anp.matmul(a2, a4)
    u = anp.matmul(a, B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade9(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    a6 = anp.matmul(a2, a4)
    a8 = anp.mtamul(a2, a6)
    u = anp.matmul(a, B[9] * a8 + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[8] * a8 + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade13(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    a6 = anp.matmul(a2, a4)
    u = anp.matmul(a, anp.matmul(a6, B[13] * a6 + B[11] * a4 + B[9] * a2) + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = anp.matmul(a6, B[12] * a6 + B[10] * a4 + B[8] * a2) + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


# Valid pade orders for algorithm 2.3.
PADE_ORDERS = (
    3,
    5,
    7,
    9,
    13,
)


# Pade approximation functions.
PADE = [
    None,
    None,
    None,
    pade3,
    None,
    pade5,
    None,
    pade7,
    None,
    pade9,
    None,
    None,
    None,
    pade13,
]


# Constants taken from table 2.3.
THETA = (
    0,
    0,
    0,
    1.495585217958292e-2,
    0,
    2.539398330063230e-1,
    0,
    9.504178996162932e-1,
    0,
    2.097847961257068,
    0,
    0,
    0,
    5.371920351148152,
)


def expm_pade(a):
    """
    Compute the matrix exponential via pade approximation.

    References:
    [0] http://eprints.ma.man.ac.uk/634/1/high05e.pdf
    [1] https://github.com/scipy/scipy/blob/v0.14.0/scipy/linalg/_expm_frechet.py#L10

    Arguments:
    a :: ndarray(N x N) - The matrix to exponentiate.
    
    Returns:
    expm_a :: ndarray(N x N) - The exponential of a.
    """
    # If the one norm is sufficiently small,
    # pade orders up to 13 are well behaved.
    scale = 0
    size = a.shape[0]
    one_norm_ = one_norm(a)
    pade_order=None
    for pade_order_ in PADE_ORDERS:
        if one_norm_ < THETA[pade_order_]:
            pade_order = pade_order_
        #ENDIF
    #ENDFOR

    # If the one norm is large, scaling and squaring
    # is required.
    if pade_order is None:
        pade_order = 13
        scale = max(0, int(anp.ceil(anp.log2(one_norm_ / THETA[13]))))
        a = a * (2 ** -scale)

    # Execute pade approximant.
    i = anp.eye(size)
    u, v = PADE[pade_order](a, i)
    r = anp.linalg.solve(-u + v, u + v)

    # Do squaring if necessary.
    for _ in range(scale):
        r = anp.matmul(r, r)

    return r


### EXPM IMPLEMENTATION VIA EIGEN DECOMPOSITION AND DIAGONALIZATION ###

def expm_eigh(h):
    """
    Compute the unitary operator of a hermitian matrix.
    U = expm(-1j * h)

    Arguments:
    h :: ndarray (N X N) - The matrix to exponentiate, which must be hermitian.
    
    Returns:
    expm_h :: ndarray(N x N) - The unitary operator of a.
    """
    eigvals, p = anp.linalg.eigh(h)
    p_dagger = anp.conjugate(anp.swapaxes(p, -1, -2))
    d = anp.exp(-1j * eigvals)
    return anp.matmul(p *d, p_dagger)


### EXPORT ###

expm = expm_pade
