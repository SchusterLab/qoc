
import numpy as np
from qoc.standard import (
                          get_annihilation_operator,
                          get_creation_operator,
                          )
import sys
from qoc.standard.functions.expm_manual import expm_pade
from scipy.sparse.linalg import expm_multiply,expm
from scipy.sparse import dia_matrix,bsr_matrix
from qoc.core.common import expm_frechet
def derivative_Expansion(HILBERT_SIZE):
    diagnol = np.arange(HILBERT_SIZE)
    up_diagnol = np.sqrt(diagnol)
    low_diagnol = np.sqrt(np.arange(1, HILBERT_SIZE + 1))
    state = np.zeros(HILBERT_SIZE)
    state[100]=1
    data = [low_diagnol, diagnol, up_diagnol]
    offsets = [-1, 0, 1]
    A = dia_matrix((data, offsets), shape=(HILBERT_SIZE, HILBERT_SIZE)).todense()
    B = dia_matrix(([low_diagnol, up_diagnol], [-1, 1]), shape=(HILBERT_SIZE, HILBERT_SIZE)).todense()
    c = np.block([[A, B], [np.zeros_like(A), A]])
    c=bsr_matrix(c).tocsc()
    state0=np.zeros_like(state)
    state=np.block([state0,state])
    state=expm_multiply(c,state)
    return
derivative_Expansion(2000)
