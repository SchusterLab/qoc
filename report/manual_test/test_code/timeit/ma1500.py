import os
os.environ['OMP_NUM_THREADS'] = '8' # set number of OpenMP threads to run in parallel
from scipy.sparse import *
from quspin.tools.misc import get_matvec_function,matvec
import numpy as np
@profile
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
def para(H,vec):
    for i in range(10000):
        matvec(H,vec)
def sci(H,vec):
    for i in range(10000):
        H.dot(vec)
scip=[]
sci_pa=[]
N_=3000
H,vec=get_H(N_,np.float64)
vec=np.complex128(vec)
para(H,vec)