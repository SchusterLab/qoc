"""
common.py - This module defines methods that are used by
multiple core functionalities.
"""

import autograd.numpy as anp
import scipy as scipy

from qoc.core.mathmethods import (interpolate_linear_set,
                                  magnus_m2,
                                  magnus_m4,
                                  magnus_m6,)
from qoc.models import (
                        InterpolationPolicy,
                        MagnusPolicy,
                        )
from qoc.standard import (
                          expm,conjugate_transpose)
import numpy as np

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
                                       magnus_policy=MagnusPolicy.M2, if_back=None, ):
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

    step_unitary = expm(magnus)
    if if_back is True:
        propagator = conjugate_transpose(step_unitary)
    else:
        propagator = step_unitary

    return magnus,propagator

def get_Hkbar(dt,Hk,H_total,approximation):
    if approximation==True:
        return Hk
    else:
        return anp.matmul(1j*scipy.linalg.expm_frechet(H_total, -1j*dt*Hk,compute_expm = False),conjugate_transpose(expm(H_total)))/dt