
from scipy.sparse import identity,csc_matrix, dia_matrix

from scipy.sparse import kron
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity)
import numpy as np
def Identity(H_size):
    return identity(H_size)

def harmonic(H_size):
    diagnol = np.arange(H_size)
    up_diagnol = np.sqrt(diagnol)
    low_diagnol = np.sqrt(np.arange(1, H_size + 1))
    a= dia_matrix(([ up_diagnol], [ 1]), shape=(H_size, H_size)).tocsc()
    a_dag=dia_matrix(([ low_diagnol], [ -1]), shape=(H_size, H_size)).tocsc()
    return a_dag,a

def transmon(w_01,anharmonicity,H_size):
    b_dag,b=harmonic(H_size=H_size)
    H0=b_dag.dot(b)
    diagnol=np.ones(H_size)
    I= dia_matrix(([ diagnol], [ 0]), shape=(H_size, H_size)).tocsc()
    H0=w_01*H0+anharmonicity/2*H0*(H0-I)
    return H0,b_dag,b
def get_control(N):
    sigmap, sigmam = harmonic(N)
    sigmap=sigmap
    sigmam=sigmam
    sigmax=sigmap+sigmam
    control=[]
    a = identity(N ** (2))
    control.append(kron(sigmax, a, format="csc"))
    for i in range(1, 2):
        control.append(kron(kron(identity(N ** i), sigmax), identity(N ** (2 - i)), format="csc"))
    control.append(kron(identity(N ** (2)), sigmax, format="csc"))
    return control
def get_int(N):
    alpha=-0.225*2*np.pi
    g=0.1*2*np.pi
    sigmap, sigmam = harmonic(N)
    sigmaz=sigmap.dot(sigmam)
    H_int=g*kron(sigmam,sigmap)+g*kron(sigmap,sigmam)
    H_anh=1/2*alpha*(sigmaz*(sigmaz-identity(N)))
    H0=kron(H_anh,identity(N**2))
    H0=H0+kron(identity(N),kron(H_anh,identity(N)))
    H0 = H0 + kron(identity(N**2), H_anh)
    H0 = H0 + kron(H_int,identity(N))
    H0 = H0 + kron(identity(N ),H_int)
    return H0

def Had(d):
    Had=np.zeros((d,d))
    Had[0][0]=1
    Had[0][1] = 1
    Had[1][0] = 1
    Had[1][1] = 1
    Had=1/np.sqrt(2)*Had
    Had_gat=Had
    for i in range(3-1):
        Had_gat=np.kron(Had_gat,Had)
    return Had_gat.reshape(d**3,d**3,1)

def get_initial(N):
    state=[]
    for i in range(N**3):
        s=np.zeros((N ** 3, 1))
        s[i]=1
        state.append(s)
    return np.array(state)

def simulation(q_number, mode):
    H_0=csc_matrix(get_int(q_number))
    H_control=get_control(q_number)

    evolution_time=0.25
    CONTROL_EVAL_COUNT = 3
    ITERATION_COUNT = 1

    Initial_state = get_initial(q_number).reshape((q_number ** 3, q_number ** 3))
    Target = Had(q_number).reshape(q_number ** 3, q_number ** 3)
    # Target = np.ones((q_number ** 3, q_number ** 3))
    COSTS = [TargetStateInfidelity(Target, cost_multiplier=1)]
    H_0 = H_0.toarray()
    for i in range(len(H_control)):
        H_control[i] = H_control[i].toarray()

    result = grape_schroedinger_discrete(H_0, H_control, CONTROL_EVAL_COUNT,
                                         COSTS, evolution_time,
                                         initial_states=Initial_state,
                                         iteration_count=ITERATION_COUNT, gradients_method=mode, expm_method="taylor")
    return result
