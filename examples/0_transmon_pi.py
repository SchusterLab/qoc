
import autograd.numpy as np
from qoc import grape_schroedinger_discrete
from qoc.standard import (TargetStateInfidelity,ForbidStates,OperatorAverage,
                          ForbidStates,
                          conjugate_transpose,
                          get_annihilation_operator,
                          get_creation_operator,
                          SIGMA_Z,
                          generate_save_file_path,ControlVariation,Adam,LBFGSB)
from scipy.sparse import dia_matrix,identity
from scipy.sparse import kron
from qutip import *
# Define the system.
def Rx(matrix,angle):
    matrix[0,0]=np.cos(angle/2)
    matrix[0,1]=-1j*np.sin(angle/2)
    matrix[1,0]=-1j*np.sin(angle/2)
    matrix[1,1]=np.cos(angle/2)
    return matrix
def pulses_der(x, paras,total_time):
    control = -2*np.pi/total_time*(paras[0]-angle/2)*np.sin(2*np.pi/total_time*x)
    for i in range(len(paras)):
        if i==len(paras)-1:
            control += -2*(i+2)*np.pi/total_time*(-paras[i])*np.sin(2*(i+2)*np.pi/total_time*x)
        else:
            control += -2*(i+2)*np.pi/total_time*(paras[i+1]-paras[i])*np.sin(2*(i+2)*np.pi/total_time*x)
    return control/total_time
def pulses(x, paras,total_time):
    control = angle/2 + (paras[0]-angle/2)*np.cos(2*np.pi/total_time*x)
    for i in range(len(paras)):
        if i==len(paras)-1:
            control += (-paras[i])*np.cos(2*(i+2)*np.pi/total_time*x)
        else:
            control += (paras[i+1]-paras[i])*np.cos(2*(i+2)*np.pi/total_time*x)
    return control/total_time
from numpy.fft import fft,fftfreq
def pulse_smoother(pulse,low_freq_lim,
                   high_freq_lim,N,total_time):
    times = np.linspace(0, total_time, N+1)
    times=np.delete(times, [len(times) - 1])
    freq = fftfreq(len(times), times[1]-times[0])
    fourier = fft(pulse)
    for i in range(len(freq)):
        if np.abs(freq[i]) > high_freq_lim:
            fourier[i]=0
        if np.abs(freq[i]) < low_freq_lim:
            fourier[i]=0
    filtered_pulse=np.real(np.fft.ifft(fourier))
    return filtered_pulse
def impose_bc(controls):
    controls[0]=pulse_smoother(controls[0],-1,
                   7/total_time,total_time_steps,total_time)
    controls[1]=pulse_smoother(controls[1],-1,
                    20/total_time,total_time_steps,total_time)
    controls[0][0]=0
    controls[0][-1]=0
    controls[1][0]=0
    controls[1][-1]=0
#     controls[1]=0*controls[0]
#     controls[1]=get_firstder(controls[0],total_time/total_time_steps)
    return controls
N_q = 4
a_q = destroy(N_q)
n_q = a_q.dag() * a_q
x_q = a_q + a_q.dag()
w_q = 0
k_q = -200e-3 * 2*np.pi  # anharmonicity/2
cross = 0e-3 * 2*np.pi
# without -1/2, strange result when change time origin
H0 = (w_q + cross) * (n_q ) + 1/2*k_q * a_q.dag()**2 * a_q**2
Hcx=a_q+a_q.dag()
H0=H0.data.toarray()
Hcx=Hcx.data.toarray()
Hcy=-a_q*1j+1j*a_q.dag()
Hcy=Hcy.data.toarray()
H_controls=[Hcx,Hcy]
total_time=5
total_time_steps=3
#Toltal number of descretized time pieces
target_states=np.zeros([N_q,N_q],dtype=complex)
angle= np.pi
target_states=Rx(target_states,angle)
Tstate_0 = np.zeros(N_q)
Tstate_0[0] = 1
Tstate_0 = target_states@Tstate_0
Tstate_1 = np.zeros(N_q)
Tstate_1[1] = 1
Tstate_1 = target_states@Tstate_1
target_states = np.array([Tstate_0])
initial_state0 = np.zeros(N_q)
initial_state0[0] = 1
initial_state1 = np.zeros(N_q)
initial_state1[1] = 1
initial_states = np.array([initial_state0])
fluc_para = 2*np.pi*1e-3*np.array([-10,-7,-4,-1,1,4,7,10])
cost1=TargetStateInfidelity(target_states=target_states, cost_multiplier= 1)
def projector_tran(dim_trans,mode,i=0):
    tran0 = np.zeros(dim_trans)
    tran0[i]=1
    tran0=dia_matrix(([tran0],[0]),shape=(dim_trans, dim_trans)).tocsc()
    if mode=="AD" or mode=="SAD":
        return tran0.toarray()
    else:
        return tran0
# cost3=ControlBandwidthMax(control_num=1,total_time_steps=total_time_steps,
#                           evolution_time=total_time,bandwidths=np.array([[-1,3/total_time],[-1,11/total_time]]),cost_multiplier=5e-3)
cost = [OperatorAverage(projector_tran(4,"SAD"), cost_multiplier=1/3)]

def robust_operator(para):
    return para*n_q.data.toarray()



robustness_set = [fluc_para, robust_operator]
initial_control = np.array([[1. ]])
def control_funcx(controls,time,i):
    return controls[i]
control_func = [control_funcx,control_funcx]
times = np.linspace(0, total_time, total_time_steps+1)
times=np.delete(times, [len(times) - 1])
paras=[-1,2,-4]
# paras=[10.49915006,  3.81964062, -1.62873995]
initial_controlx= pulses(times,paras,total_time)
initial_controly= -pulses_der(times,paras,total_time)/(2*k_q)
initial_control=[initial_controlx,initial_controly]
result = grape_schroedinger_discrete(H0, H_controls, total_time_steps,
                                         cost, total_time,
                                         initial_states, initial_controls = initial_control,control_func= control_func ,log_iteration_step = 1, min_error=1e-1,
                                         iteration_count=1, gradients_method="SAD", expm_method="taylor",impose_control_conditions=impose_bc, )
