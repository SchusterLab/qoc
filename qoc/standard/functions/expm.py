"""
expm.py - a module for all things e^M
"""

import scipy.sparse.linalg
from scipy.sparse import isspmatrix
import importlib
from autograd.extend import Box

def expm(A, vector, method="pade", gradients_method="AD" ):
    if gradients_method == "AD":
        globals()["np"] = importlib.import_module("autograd.numpy")
    else:
        globals()["np"] = importlib.import_module("numpy")
    if method == "pade":
        exp_matrix = expm_pade(A)
        vector = np.matmul(exp_matrix, vector)
    if method == "taylor":
        vector = expm_taylor(A, vector, )
    return vector

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
    return np.max(np.sum(np.abs(a), axis=0))
    

def pade3(a, i):
    a2 = np.matmul(a, a)
    u = np.matmul(a, B[2] * a2) + B[1] * a
    v = B[2] * a2 + B[0] * i
    return u, v


def pade5(a, i):
    a2 = np.matmul(a, a)
    a4 = np.matmul(a2, a2)
    u = np.matmul(a, B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade7(a, i):
    a2 = np.matmul(a, a)
    a4 = np.matmul(a2, a2)
    a6 = np.matmul(a2, a4)
    u = np.matmul(a, B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade9(a, i):
    a2 = np.matmul(a, a)
    a4 = np.matmul(a2, a2)
    a6 = np.matmul(a2, a4)
    a8 = np.mtamul(a2, a6)
    u = np.matmul(a, B[9] * a8 + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[8] * a8 + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


def pade13(a, i):
    a2 = np.matmul(a, a)
    a4 = np.matmul(a2, a2)
    a6 = np.matmul(a2, a4)
    u = np.matmul(a, np.matmul(a6, B[13] * a6 + B[11] * a4 + B[9] * a2) + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = np.matmul(a6, B[12] * a6 + B[10] * a4 + B[8] * a2) + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
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
    pade_order = None
    one_norm_ = one_norm(a)
    for pade_order_ in PADE_ORDERS:
        if one_norm_ < THETA[pade_order_]:
            pade_order = pade_order_
        #ENDIF
    #ENDFOR
    # If the one norm is large, scaling and squaring
    # is required.
    if pade_order is None:
        pade_order = 13
        scale = max(0, int(np.ceil(np.log2(one_norm_ / THETA[13]))))
        a = a * (2 ** -scale)

    # Execute pade approximant.
    i = np.eye(size)
    u, v = PADE[pade_order](a, i)
    r = np.linalg.solve(-u + v, u + v)

    # Do squaring if necessary.
    for _ in range(scale):
        r = np.matmul(r, r)

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
    eigvals, p = np.linalg.eigh(h)
    p_dagger = np.conjugate(np.swapaxes(p, -1, -2))
    d = np.exp(-1j * eigvals)
    return np.matmul(p *d, p_dagger)


### EXPORT ###
def _exact_inf_norm(A):
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    else:
        import numpy as np
        if isinstance(A, Box):
            return np.linalg.norm(A._value, np.inf)
        else:
            return np.linalg.norm(A, np.inf)

def trace(A):
    """
    Args:
    A :: numpy.ndarray - matrix
    Returns:
    trace of A
    """
    if scipy.sparse.isspmatrix(A):
        return A.diagonal().sum()
    else:
        return np.trace(A)

def ident_like(A):
    """
    Args:
    A :: numpy.ndarray - matrix
    Returns:
    Identity matrix which has same dimension as A
    """
    if scipy.sparse.isspmatrix(A):
        return scipy.sparse.construct.eye(A.shape[0], A.shape[1],
                                          dtype=A.dtype, format=A.format)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
def gamma_fa(n):
    u = 1.11e-16
    u=np.sqrt(2)*2*u/(1-2*u)
    return n*u/(1-n*u)
def beta(norm,m,n,tol):
    beta=gamma_fa(m+1)
    r = 1
    for i in range(1,m):
        r=r*norm/i
        g = gamma_fa(i*(n+2)+m+2)
        beta = beta+g*r
        if beta>tol:
            return 0,0
    return beta,r

def taylor_term(i,norm,term):
    return term*norm/i
def error(norm_B,m,s,n,R_m,tol):
    tr = R_m
    rd=beta(norm_B,m,n,tol)[0]
    rd=np.power((1+rd+tr),s)-np.power((1+tr),s)
    tr=tr*s
    tr=tr*((1-np.power(tr,s))/(1-tr))
    return tr+rd
def weaker_error(beta,R_m,s):
    tr = R_m
    rd=beta
    rd = np.power((1 + rd + tr), s) - np.power((1 + tr), s)
    tr = tr * s
    tr = tr * ((1 - np.power(tr, s)) / (1 - tr))
    return tr+rd
def residue_norm(m,norm_B,term):
    R_m=term
    for i in range(m+2,1000):
        term=term*norm_B/i
        R_m=R_m+term
        if term<1e-15:
            break
    return R_m

def choose_ms(norm_A,d,tol):
    no_solution=True
    for i in range(1,int(np.ceil(norm_A))+1):
        if no_solution == False:
            break
        norm_B = norm_A / i
        l=int(np.floor(norm_B))
        beta_factor,last_term=beta(norm_B,l,d,tol)
        if beta_factor==0:
            continue
        lower_bound = i*(beta_factor)
        if lower_bound>tol:
            continue
        tr_first_term=norm_B
        m_pass_lowbound=False
        for j in range(1,100):
            if j>l:
                last_term=last_term*norm_B/j
                if last_term<1e-15:
                    break
                beta_factor=beta_factor+gamma_fa(j*(d+2)+2)*last_term
            if m_pass_lowbound == False:
                tr_first_term = tr_first_term * (norm_B / (j + 1))
                if i *tr_first_term + lower_bound > tol:
                    continue
                else:
                    R_m = residue_norm(j, norm_B, tr_first_term)
                    m_pass_lowbound = True
            if m_pass_lowbound == True:
                if weaker_error(beta_factor,R_m,i)>tol:
                    R_m = R_m - tr_first_term
                    tr_first_term = tr_first_term * norm_B / (j + 1)

                    continue
                else:
                    total_error=error(norm_B,j,i,d,R_m,tol)
                    R_m = R_m - tr_first_term
                    tr_first_term = tr_first_term * norm_B / (j + 1)
                    if total_error<tol:
                        no_solution = False
                        s=i
                        m=j
                        break

    if no_solution==False:
        return s,m
    if no_solution == True:
        raise ValueError("please lower the error tolerance ")

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
@profile
def expm_taylor(A, B, d=5, tol=1e-8):
    """
    A helper function.
    """
    if tol is None:
        tol =1e-8
    norm_A = _exact_inf_norm(A)
    s,m=choose_ms(norm_A,d,tol)
    F=B
    for i in range(int(s)):
        for j in range(m):
            coeff = s*(j+1)
            B = np.matmul(A,B)/coeff
            F = F + B
        B = F
    return F




def expmat_der_vec_mul(A, E, tol, state, expm_method, gradients_method):
    """
        Calculate the action of propagator derivative.
        First we construct auxiliary matrix and vector. Then use expm_multiply function.
        Arg:
        A :: numpy.ndarray - Total Hamiltonian
        E :: numpy.ndarray - Control Hamiltonian
        state :: numpy.ndarray
        Returns:
        numpy.ndarray,numpy.ndarray - vector for gradient calculation, updated state
        """
    state=np.complex128(state)

    d = len(state[0])
    if tol==None:
        tol=1e-8
    control_number = len(E)
    final_matrix = []
    for i in range(control_number+1):
        final_matrix.append([])
    if isspmatrix(A) == False:
        for i in range(control_number + 1):
            raw_matrix = []
            if i == 0:
                raw_matrix = raw_matrix + [A]
            else:
                raw_matrix = raw_matrix + [np.zeros_like(A)]
            for j in range(1, control_number + 1):
                if j == i and i != 0:
                    raw_matrix = raw_matrix + [A]
                elif j == control_number and j != i:
                    raw_matrix = raw_matrix + [E[i]]
                else:
                    raw_matrix = raw_matrix + [np.zeros_like(A)]
            final_matrix[i] = raw_matrix
        final_matrix = np.block(final_matrix)
    else:
        for i in range(control_number+1):
            raw_matrix = []
            if i == 0:
                raw_matrix=raw_matrix+[A]
            else:
                raw_matrix = raw_matrix+[None]
            for j in range(1,control_number+1):
                if j == i and i != 0:
                    raw_matrix = raw_matrix+[A]
                elif j == control_number  and j!=i:
                    raw_matrix = raw_matrix+[E[i]]
                else:
                    raw_matrix = raw_matrix+[None]
            final_matrix[i] = raw_matrix
        final_matrix=scipy.sparse.bmat(final_matrix)
    state0 = np.zeros_like(state)
    for i in range(control_number):
        state = np.block([state0, state])
    state = state.transpose()
    state = expm(final_matrix, state , expm_method, gradients_method )
    states = []
    for i in range(control_number):
        states.append(state[d*i:d*(i+1)].transpose())
    return np.array(states), state[control_number*d:d*(control_number+1)].transpose()