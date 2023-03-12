
from scipy.sparse import kron,identity,csc_matrix
from qoc.standard.constants import harmonic,transmon,coherent_state,Identity
from scipy.sparse import kron
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,ControlVariation)
from qoc.standard import Adam
from qoc.models.operationpolicy import OperationPolicy
import numpy as np
def get_control(N):
    sigmap, sigmam = harmonic(N)
    sigmap=sigmap
    sigmam=sigmam
    sigmax=sigmap+sigmam
    control=[]
    if N==1:
        control.append(kron(sigmax, identity(N ** (3))))
        return control
    else:
        a=identity(N**(3))
        control.append(kron(sigmax,a,format="csc"))
        for i in range(1,3):
            control.append(kron(kron(identity(N**i),sigmax), identity(N ** (3-i)),format="csc"))
        control.append(kron(identity(N**(3)),sigmax,format="csc"))
    return control
def get_int(N):
    alpha=-0.225
    sigmap, sigmam = harmonic(N)
    sigmaz=sigmap.dot(sigmam)
    H_int=kron(sigmam,sigmap)+kron(sigmap,sigmam)
    H0=0
    H_anh=1/2*alpha*(sigmaz*(sigmaz-identity(N)))
    H0=kron(H_anh,identity(N**3))
    H0=H0+kron(identity(N),kron(H_anh,identity(N**2)))
    H0 = H0 + kron(identity(N**2), kron(H_anh, identity(N )))
    H0 = H0 + kron(identity(N**3), H_anh)
    H0 = H0 + kron(H_int,identity(N**2))
    H0 = H0 + kron(identity(N), kron(H_int, identity(N)))
    H0 = H0 + kron(identity(N ** 2),H_int)
    return H0

def Had(d):
    Had=np.zeros((d,d))
    Had[0][0]=1
    Had[0][1] = 1
    Had[1][0] = 1
    Had[1][1] = 1
    Had=1/np.sqrt(2)*Had
    Had_gat=Had
    for i in range(4-1):
        Had_gat=np.kron(Had_gat,Had)
    return Had_gat.reshape(d**4,d**4,1)

def control_H(control,H_control):
    H=0
    for i in range(len(control)):
        H=H+control[i]*H_control[i]
    return H
def get_initial(N):
    state=[]
    for i in range(N**4):
        s=np.zeros((N ** 4, 1))
        s[i]=1
        state.append(s)
    return np.array(state)

def simulation(q_number,max_con,mode):
    H_0=csc_matrix(get_int(q_number))
    H_control=get_control(q_number)
    hamiltonian = lambda controls, time: (H_0
                                          + control_H(controls,H_control))
    Initial_state=get_initial(q_number)
    Target=Had(q_number)
    CONTROL_COUNT = 4
    evolution_time=0.25
    CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = 1+ 1
    ITERATION_COUNT = 1

    max_control_norms=max_con*np.ones(CONTROL_COUNT)

    COSTS = [TargetStateInfidelity(Target,cost_multiplier=1)]

    # Define output.
    LOG_ITERATION_STEP = 1

    manual_parameter={"control_hamiltonian":H_control,"manual_gradient_mode":True,"tol":1e-8}

    def Impose(control):
        control[0] = control[control.shape[0] - 1] = 0
        return control
    if mode=="AG":
        manual_parameter={"control_hamiltonian":H_control,"manual_gradient_mode":True,"tol":1e-8}
        Target = Had(q_number)
        COSTS = [TargetStateInfidelity(Target, cost_multiplier=1)]
    else:
        Initial_state = get_initial(q_number).reshape((q_number**4,q_number**4))
        Target = Had( q_number).reshape(q_number**4, q_number**4)
        COSTS = [TargetStateInfidelity(Target, cost_multiplier=1)]
        H_0 = H_0.toarray()
        for i in range(len(H_control)):
            H_control[i] = H_control[i].toarray()
        manual_parameter = {"control_hamiltonian": H_control, "manual_gradient_mode": False, "tol": 1e-2}
    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                             COSTS, evolution_time, hamiltonian,
                                             Initial_state, SYSTEM_EVAL_COUNT,
                                             complex_controls=False,

                                             iteration_count=ITERATION_COUNT,
                                             log_iteration_step=LOG_ITERATION_STEP, min_error=0.001,
                                             max_control_norms=max_control_norms,
                                             impose_control_conditions=Impose,
                                            manual_parameter=manual_parameter,
                                             save_iteration_step=1,
                                             )
    return result
