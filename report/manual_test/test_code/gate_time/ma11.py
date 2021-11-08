from qoc.standard.constants import harmonic, transmon, coherent_state, Identity
from scipy.sparse import kron
import numpy as np
import sys
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity, ControlNorm, ControlVariation, generate_save_file_path,
                          ForbidStates)
from qoc.standard import Adam
from scipy import signal
from qutip import coherent
from qoc.models.operationpolicy import OperationPolicy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path, )
from memory_profiler import profile
import numpy as np

from qutip import (qsave, qload)
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix


import numpy as np
from scipy.sparse import kron,identity,csc_matrix
from qoc.standard.constants import harmonic,transmon,coherent_state,Identity
from qoc.standard import (TargetStateInfidelity,

                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path,)
from qoc.standard.constants import harmonic,transmon,coherent_state,Identity
from scipy.sparse import kron
import numpy as np
import sys
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,ControlNorm,ControlVariation,generate_save_file_path,
                          ForbidStates)
from qoc.standard import Adam
from scipy import signal
from qutip import coherent
from qoc.models.operationpolicy import OperationPolicy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['NUMEXPR_MAX_THREADS']='32'
def get_control(N):
    sigmap, sigmam = harmonic(2)
    sigmap=sigmap
    sigmam=sigmam
    sigmax=sigmap+sigmam
    sigmay=-1j*sigmap+1j*sigmam
    control=[]
    if N==1:
        control.append(kron(sigmax, identity(2 ** (N - 1))))
        control.append(kron(sigmay, identity(2 ** (N - 1))))
        return control
    else:
        a=identity(2**(N-1))
        control.append(kron(sigmax,a,format="csc"))
        control.append(kron(sigmay, identity(2 ** (N - 1)),format="csc"))
        for i in range(1,N-1):
            control.append(kron(kron(identity(2**i),sigmax), identity(2 ** (N - 1-i)),format="csc"))
            control.append(kron(kron(identity(2 ** i), sigmay), identity(2 ** (N - 1 - i)),format="csc"))
        control.append(kron(identity(2**(N-1)),sigmax,format="csc"))
        control.append(kron(identity(2**(N-1)),sigmay,format="csc"))
    return control
def get_int(N):
    sigmap, sigmam = harmonic(2)
    sigmaz=sigmap.dot(sigmam)
    H0=0
    SIGMAZ=kron(sigmaz,sigmaz)
    H0=H0+kron(SIGMAZ,identity(2**(N-2)))+kron(identity(2**(N-2)),SIGMAZ)
    for i in range(1,N-2):
        H0=H0+kron(kron(identity(2**i),SIGMAZ),identity(2 ** (N - 2 - i)))
    return H0

def Had(d,n):
    omega=np.exp(2j*np.pi/d)
    Had = 1/np.sqrt(d) * np.array([[((omega) ** (i*j))
                                      for i in range(d)]
                                     for j in range(d)])
    Had_gat=Had
    for i in range(n-1):
        Had_gat=np.kron(Had_gat,Had)
    return Had_gat.reshape(d**n,d**n,1)

def control_H(control,H_control):
    H=0
    for i in range(len(control)):
        H=H+control[i]*H_control[i]
    return H
def get_initial(N):
    state=[]
    for i in range(2**N):
        s=np.zeros((2 ** N, 1))
        s[i]=1
        state.append(s)
    return np.array(state)

def simulation(q_number,max_con,initial):
    H_0=csc_matrix(get_int(q_number))
    H_control=get_control(q_number)
    hamiltonian = lambda controls, time: (H_0
                                          + control_H(controls,H_control))
    Initial_state=get_initial(q_number)
    Target=Had(2,q_number)
    CONTROL_COUNT = 2*q_number
    evolution_time=0.2
    CONTROL_EVAL_COUNT = SYSTEM_EVAL_COUNT = 1+ 1
    ITERATION_COUNT = 1

    max_control_norms=max_con*np.ones(CONTROL_COUNT)
    cost_first = ControlVariation(control_count=CONTROL_COUNT,
                             control_eval_count=CONTROL_EVAL_COUNT,
                             cost_multiplier=0.01,
                             max_control_norms=max_control_norms*0.2,
                             order=1)
    cost_second = ControlVariation(control_count=CONTROL_COUNT,
                                  control_eval_count=CONTROL_EVAL_COUNT,
                                  cost_multiplier=0.001,
                                  max_control_norms=max_control_norms*0.05,
                                  order=1)

    COSTS = [TargetStateInfidelity(Target,cost_multiplier=1)]

    # Define output.
    LOG_ITERATION_STEP = 1

    manual_parameter={"control_hamiltonian":H_control,"manual_gradient_mode":True,"tol":1e-8}
    optimizer = Adam(beta_1=0.9, beta_2=0.999, clip_grads=None,
                     epsilon=1e-8, learning_rate=1e-3,
                     learning_rate_decay=None, operation_policy=OperationPolicy.CPU,
                     scale_grads=None)

    def Impose(control):
        control[0] = control[control.shape[0] - 1] = 0
        return control

    result = grape_schroedinger_discrete(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                             COSTS, evolution_time, hamiltonian,
                                             Initial_state, SYSTEM_EVAL_COUNT,
                                             complex_controls=False,
                                             initial_controls=initial,
                                             iteration_count=ITERATION_COUNT,
                                             log_iteration_step=LOG_ITERATION_STEP, min_error=0.001,
                                             max_control_norms=max_control_norms,
                                             impose_control_conditions=Impose,
                                            manual_parameter=manual_parameter,
                                             save_iteration_step=1,
                                             )
    return result

result=simulation(11,0.5,None)

