import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.sputils import is_pydata_spmatrix
from scipy.sparse import *
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
        s=np.ceil(overnorm(A))
    else:
        while(1):
            tol_power = np.ceil(np.log10(tol))
            norm_A = overnorm(A)/s
            norm_b= norm_state(b)
            max_term_notation=np.floor(norm_A)
            max_term=1
            for i in range(1,int(max_term_notation)):
                max_term=max_term*norm_A/i
                max_power = np.ceil(np.log10(max_term))
                if max_power>30:
                    break
            max_power = np.ceil(np.log10(max_term))
            if max_power-16<=tol_power:
                break
            s=s+1
    return s
def overnorm(A):
    if A.dtype==np.complex128:
        return _exact_inf_norm(A)
    else:
        return norm_two(A)
def norm_two(A):
    if sci.sparse.isspmatrix(A):
        A=csr_matrix(A).conjugate().transpose()
        return np.sqrt(abs(eigs(A=A.dot(A),k=1,which='LM',return_eigenvectors=False)[0]))
    else:
        return np.linalg.norm(A)
def norm_state(A):
    return np.linalg.norm(A)
@profile
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
    j=0
    while(1):
        eta = np.exp(mu / float(s))
        coeff = s*(j+1)
        B =  A.dot(B)/coeff
        c2 = overnorm(B)
        F = F + B
        total_norm=norm_state(F)
        if c2/total_norm<tol:
            m=j+1
            break
        j=j+1
    F = eta * F
    B = F
    for i in range(1,int(s)):
        eta = np.exp(mu / float(s))
        for j in range(50):
            coeff = s*(j+1)
            B =  A.dot(B)/coeff
            c2 =norm_state(B)
            F = F + B
            total_norm=norm_state(F)
            if c2/total_norm<tol:
                m=j+1
        F = eta*F
        B = F
    return F
import numpy as np
def get_creation_operator(size,tp):
    return np.diag(np.sqrt(np.arange(1, size),dtype=tp), k=-1)
def get_annihilation_operator(size,tp):
    return np.diag(np.sqrt(np.arange(1, size),dtype=tp), k=1)
def get_H(dim,tp):
    HILBERT_SIZE=dim
    Q_dim=6
    g=0.1*2*np.pi
    a_dag = get_creation_operator(HILBERT_SIZE,tp)
    a = get_annihilation_operator(HILBERT_SIZE,tp)
    b_dag=get_creation_operator(Q_dim,tp)
    b=get_annihilation_operator(Q_dim,tp)
    A=np.kron(a,np.identity(Q_dim))
    A_dag=np.kron(a_dag,np.identity(Q_dim))
    B=np.kron(np.identity(HILBERT_SIZE),b)
    B_dag=np.kron(np.identity(HILBERT_SIZE),b_dag)
    H0=g*(np.kron(a_dag,b)+np.kron(a,b_dag))
    H=H=csr_matrix(-1j*(H0+0.5*2*np.pi*(A+A_dag+B+B_dag+1j*(B-B_dag+A-A_dag))))
    vec=1j*1/np.sqrt(HILBERT_SIZE*Q_dim)*np.ones(HILBERT_SIZE*Q_dim)
    return H,vec
def sci(H,vec):
    for i in range(100):
        H.dot(vec)
scip=[]
sci_pa=[]
N=3000
H,vec=get_H(N,np.float64)
vec=np.complex128(vec)
expm_multiply(H,vec)