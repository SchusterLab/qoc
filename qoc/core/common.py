"""
common.py - This module defines methods that are used by
multiple core functionalities.
"""
from qoc.core.mathmethods import (interpolate_linear_set,
                                  magnus_m2,
                                  magnus_m4,
                                  magnus_m6,)
from qoc.models import (
                        InterpolationPolicy,
                        MagnusPolicy,
                        )
from qoc.standard import (
                          conjugate_transpose)
from qoc.standard.functions.expm_manual import expm_pade
import numpy as np
import scipy.linalg
import gc
def clip_control_norms(controls, max_control_norms):
    """
    Me: I need the entry-wise norms of the column entries of my
        control array to each be scaled to a fixed
        maximum norm if they exceed that norm
    Barber: Say no more fam

    Arguments:
    controls
    max_control_norms

    Returns: None
    """
    for i, max_control_norm in enumerate(max_control_norms):
        control = controls[:, i]
        control_norm = np.abs(control)
        offending_indices = np.nonzero(np.less(max_control_norm, control_norm))
        offending_control_points = control[offending_indices]
        # Rescale the offending points to `max_control_norm`.
        resolved_control_points = ((offending_control_points / control_norm[offending_indices])
                                   * max_control_norm)
        control[offending_indices] = resolved_control_points
    #ENDFOR


def gen_controls_cos(complex_controls, control_count, control_eval_count,
                     evolution_time, max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a cosine function.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    max_control_norms

    periods

    Returns:
    controls
    """
    period = np.divide(control_eval_count, periods)
    b = np.divide(2 * np.pi, period)
    controls = np.zeros((control_eval_count, control_count))

    # Create a wave for each control over all time
    # and add it to the controls.
    for i in range(control_count):
        # Generate a cosine wave about y=0 with amplitude
        # half of the max.
        max_norm = max_control_norms[i]
        _controls = (np.divide(max_norm, 2)
                   * np.cos(b * np.arange(control_eval_count)))
        # Replace all controls that have zero value
        # with small values.
        small_norm = max_norm * 1e-1
        _controls = np.where(_controls, _controls, small_norm)
        controls[:, i] = _controls
    #ENDFOR

    # Mimic the cosine fit for the imaginary parts and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / np.sqrt(2)

    return controls

def gen_controls_white(complex_controls, control_count, control_eval_count,
                      evolution_time, max_control_norms, periods=10.):
    """
    Create a discrete control set of random white noise.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    max_control_norms

    periods

    Returns:
    controls
    """
    controls = np.zeros((control_eval_count, control_count))

    # Make each control a random distribution of white noise.
    for i in range(control_count):
        max_norm = max_control_norms[i]
        stddev = max_norm/5.0
        control = np.random.normal(0, stddev, control_eval_count)
        controls[:, i] = control
    #ENDFOR

    # Mimic the white noise for the imaginary parts, and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / np.sqrt(2)

    return controls


def gen_controls_flat(complex_controls, control_count, control_eval_count,
                      evolution_time, max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a flat line with small amplitude.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    max_control_norms

    periods

    Returns:
    controls
    """
    controls = np.zeros((control_eval_count, control_count))

    # Make each control a flat line for all time.
    for i in range(control_count):
        max_norm = max_control_norms[i]
        small_norm = max_norm * 1e-1
        control = np.repeat(small_norm, control_eval_count)
        controls[:, i] = control
    #ENDFOR

    # Mimic the flat line for the imaginary parts, and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / np.sqrt(2)

    return controls


_NORM_TOLERANCE = 1e-10
def initialize_controls(complex_controls,
                        control_count,
                        control_eval_count, evolution_time,
                        initial_controls, max_control_norms):
    """
    Sanitize `initial_controls` with `max_control_norms`.
    Generate both if either was not specified.

    Arguments:
    complex_controls
    control_count
    control_eval_count
    evolution_time
    initial_controls
    max_control_norms

    Returns:
    controls
    max_control_norms
    """
    if max_control_norms is None:
        max_control_norms = np.ones(control_count)

    if initial_controls is None:
        controls = gen_controls_flat(complex_controls, control_count, control_eval_count,
                                     evolution_time, max_control_norms)
    else:
        # Check that the user-specified controls match the specified data type.
        if complex_controls:
            if not np.iscomplexobj(initial_controls):
                raise ValueError("The program expected that the initial_controls specified by "
                                 "the user conformed to complex_controls, but "
                                 "the program found that the initial_controls were not complex "
                                 "and complex_controls was set to True.")
        else:
            if np.iscomplexobj(initial_controls):
                raise ValueError("The program expected that the initial_controls specified by "
                                 "the user conformed to complex_controls, but "
                                 "the program found that the initial_controls were complex "
                                 "and complex_controls was set to False.")

        # Check that the user-specified controls conform to max_control_norms.
        for control_step, step_controls in enumerate(initial_controls):
            if not (np.less_equal(np.abs(step_controls), max_control_norms + _NORM_TOLERANCE).all()):
                raise ValueError("The program expected that the initial_controls specified by "
                                 "the user conformed to max_control_norms, but the program "
                                 "found a conflict at initial_controls[{}]={} and "
                                 "max_control_norms={}."
                                 "".format(control_step, step_controls, max_control_norms))
        #ENDFOR
        controls = initial_controls

    return controls, max_control_norms


def slap_controls(complex_controls, controls, controls_shape,):
    """
    Reshape and transform controls in optimizer format
    to controls in cost function format.

    Arguments:
    complex_controls :: bool - whether or not the controls in cost function
         format are complex
    controls :: ndarray (2 * controls_size if COMPLEX else controls_size)
        - the controls in optimizer format
    controls_shape :: tuple(int) -

    Returns:
    controls :: ndarray (controls_shape)- the controls in cost function format
    """
    # Transform the controls to C if they are complex.
    if complex_controls:
        real, imag = np.split(controls, 2)
        controls = real + 1j * imag
    # Reshape the controls.
    controls = np.reshape(controls, controls_shape)

    return controls


def strip_controls(complex_controls, controls):
    """
    Reshape and transform controls in cost function format
    to controls in optimizer format.

    Arguments:
    complex_controls :: bool - whether or not the controls in cost function
        format are complex
    controls :: ndarray (controls_shape) - the controls in cost function format

    Returns:
    controls :: ndarray (2 * controls_size if COMPLEX else controls_size)
        - the controls in optimizer format
    """
    # Flatten the controls.
    controls = np.ravel(controls)
    # Transform the controls to R2 if they are complex.
    if complex_controls:
        controls = np.hstack((np.real(controls), np.imag(controls)))

    return controls

def interpolate_tran(control_eval_times,controls,dt):
    _M2_C1 = 0.5
    controls_=[]
    for j in range(len(controls)-1):
        t1 = dt * (_M2_C1+j)
        controls_.append(interpolate_linear_set(t1, control_eval_times, controls))
    return controls_

def gradient_trans(gradient,control_eval_times,dt):
    grads=np.zeros((len(gradient)+1,len(gradient[0])))
    time=intermidiate_time(dt,len(control_eval_times))
    for i in range(len(control_eval_times)):
        if i ==0:
            grads[i]=(gradient[i]*((control_eval_times[i+1]-time[i])/(control_eval_times[i+1]-control_eval_times[i])))
        elif i ==len(control_eval_times)-1:
            grads[i]=(gradient[i-1] * (
                        (-control_eval_times[i-1]+time[i-1]) / (control_eval_times[i] - control_eval_times[i-1])))
        else:
            grads[i]=(gradient[i]*((control_eval_times[i+1]-time[i])/(control_eval_times[i+1]-control_eval_times[i]))
                         +gradient[i-1] * ((-control_eval_times[i-1]+time[i-1]) / (control_eval_times[i] - control_eval_times[i-1])))
    return grads

def intermidiate_time(dt,system_eval_count):
    _M2_C1 = 0.5
    time=[]
    for j in range(system_eval_count-1):
        time.append(dt * (_M2_C1+j))
    return time


def get_magnus(dt, hamiltonian,
                                        time,
                                       control_eval_times=None,
                                       controls=None,
                                       interpolation_policy=InterpolationPolicy.LINEAR,
                                       magnus_policy=MagnusPolicy.M2, if_back=None, if_magnus=None):
    """
    Use the exponential series method via magnus expansion to evolve the state vectors
    to the next time step under the schroedinger equation for time-discrete controls.
    Magnus expansions are implemented using the methods described in
    https://arxiv.org/abs/1709.06483.

    Arguments:
    dt
    hamiltonian
    states
    time

    control_eval_times
    controls
    interpolation_policy
    magnus_policy

    Returns:
    states
    """
    # Choose an interpolator.
    if interpolation_policy == InterpolationPolicy.LINEAR:
        interpolate = interpolate_linear_set
    else:
        raise NotImplementedError("The interpolation policy {} "
                                  "is not yet supported for this method."
                                  "".format(interpolation_policy))

    # Choose a control interpolator.
    if controls is not None and control_eval_times is not None:
        interpolate_controls = interpolate
    else:
        interpolate_controls = lambda x, xs, ys: None

    # Construct a function to interpolate the hamiltonian
    # for all time.
    def get_hamiltonian(time_):
        controls_ = interpolate_controls(time_, control_eval_times, controls)
        hamiltonian_ = hamiltonian(controls_, time_)
        return -1j * hamiltonian_

    if magnus_policy == MagnusPolicy.M2:
        magnus = magnus_m2(get_hamiltonian, dt, time)
    elif magnus_policy == MagnusPolicy.M4:
        magnus = magnus_m4(get_hamiltonian, dt, time)
    elif magnus_policy == MagnusPolicy.M6:
        magnus = magnus_m6(get_hamiltonian, dt, time)
    else:
        raise ValueError("Unrecognized magnus policy {}."
                         "".format(magnus_policy))
    # ENDIF
    if if_magnus==True:
        return magnus
    else:
        step_unitary = expm_pade(magnus)
        if if_back is True:
            propagator = conjugate_transpose(step_unitary)
        else:
            propagator = step_unitary
        return propagator

def get_Hkbar(dt,Hk,H_total,propagator,approximation):
    if approximation==True:
        return Hk
    else:
        return np.matmul(1j*expm_frechet(H_total, -1j*dt*Hk,compute_expm = False,check_finite=False),propagator)/dt






def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    if check_finite:
        A = np.asarray_chkfinite(A)
        E = np.asarray_chkfinite(E)
    else:
        A = np.asarray(A)
        E = np.asarray(E)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    if method is None:
        method = 'SPS'
    if method == 'SPS':
        expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError('Unknown implementation %s' % method)
    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        return expm_frechet_AE


def expm_frechet_block_enlarge(A, E):
    """
    This is a helper function, mostly for testing and profiling.
    Return expm(A), frechet(A, E)
    """
    n = A.shape[0]
    M = np.vstack([
        np.hstack([A, E]),
        np.hstack([np.zeros_like(A), A])])
    expm_M = scipy.linalg.expm(M)
    return expm_M[:n, :n], expm_M[:n, n:]


"""
Maximal values ell_m of ||2**-s A|| such that the backward error bound
does not exceed 2**-53.
"""
ell_table_61 = (
        None,
        # 1
        2.11e-8,
        3.56e-4,
        1.08e-2,
        6.49e-2,
        2.00e-1,
        4.37e-1,
        7.83e-1,
        1.23e0,
        1.78e0,
        2.42e0,
        # 11
        3.13e0,
        3.90e0,
        4.74e0,
        5.63e0,
        6.56e0,
        7.52e0,
        8.53e0,
        9.56e0,
        1.06e1,
        1.17e1,
        )


# The b vectors and U and V are copypasted
# from scipy.sparse.linalg.matfuncs.py.
# M, Lu, Lv follow (6.11), (6.12), (6.13), (3.3)

def _diff_pade3(A, E, ident):
    b = (120., 60., 12., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    U = A.dot(b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
    Lv = b[2]*M2
    return U, V, Lu, Lv


def _diff_pade5(A, E, ident):
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[5]*M4 + b[3]*M2) +
            E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade7(A, E, ident):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade9(A, E, ident):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    A8 = np.dot(A4, A4)
    M8 = np.dot(A4, M4) + np.dot(M4, A4)
    U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def expm_frechet_algo_64(A, E):
    n = A.shape[0]
    s = None
    ident = np.identity(n)
    A_norm_1 = scipy.linalg.norm(A, 1)
    m_pade_pairs = (
            (3, _diff_pade3),
            (5, _diff_pade5),
            (7, _diff_pade7),
            (9, _diff_pade9))
    for m, pade in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break
    if s is None:
        # scaling
        s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
        del A_norm_1
        A = A * 2.0 ** -s
        E = E * 2.0 ** -s
        # pade order 13
        A2 = np.dot(A, A)
        M2 = np.dot(A, E) + np.dot(E, A)
        A4 = np.dot(A2, A2)
        M4 = np.dot(A2, M2) + np.dot(M2, A2)
        A6 = np.dot(A2, A4)
        M6 = np.dot(A4, M2) + np.dot(M4, A2)
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
             1187353796428800., 129060195264000., 10559470521600.,
             670442572800., 33522128640., 1323241920., 40840800., 960960.,
             16380., 182., 1.)
        W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
        W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        del ident
        gc.collect()
        A2=b[8] * A2
        A2=A2+b[10] * A4
        del A4
        gc.collect()
        Z1=b[12] * A6+A2
        del A2#10
        gc.collect()
        W = np.dot(A6, W1) + W2
        del W2
        gc.collect()
        V = np.dot(A6, Z1) + Z2
        del Z2
        gc.collect()
        U = np.dot(A, W)
        E = np.dot(E, W)
        del W#11
        gc.collect()
        Lw1 =  b[9] * M2
        Lw1=Lw1+b[11] * M4
        Lw1=Lw1+b[13] * M6
        Lw1=np.dot(A6, Lw1,Lw1)+ np.dot(M6, W1,W1)
        del W1
        gc.collect()
        M6=b[7] * M6
        M4=b[5] * M4
        M2=b[3] * M2

        Lw = Lw1+M2+M4+M6
        del Lw1
        gc.collect()
        M6=M6/b[7]
        M4=M4/b[5]
        M2=M2/b[3]
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        M2=b[2] * M2
        M2=b[4] * M4+M2
        M2 = b[6] * M6 +M2
        Lz2 = M2#13
        del M2,M4
        gc.collect()
        Lu = np.dot(A, Lw) +E
        del E,Lw,A
        gc.collect()
        Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
        del A6,M6,Z1,Lz2,Lz1
        gc.collect()
        lu_piv = scipy.linalg.lu_factor(-U + V)
        R = scipy.linalg.lu_solve(lu_piv, U + V)
        L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
        del lu_piv,Lu,Lv,U,V
        gc.collect()
        # squaring
        for k in range(s):
            L = np.dot(R, L) + np.dot(L, R)
        return  L
    # factor once and solve twice
    lu_piv = scipy.linalg.lu_factor(-U + V)
    R = scipy.linalg.lu_solve(lu_piv, U + V)
    L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
    # squaring
    for k in range(s):
        L = np.dot(R, L) + np.dot(L, R)
    return  L
