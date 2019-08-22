"""
common.py - a module to define methods used by multiple
core functionalities
"""

import numpy as np

from qoc.models.dummy import Dummy
from qoc.standard import(complex_to_real_imag_flat,
                         real_imag_to_complex_flat)

def clip_params(max_param_norms, params):
    """
    Me: I need a little taken off the top.
    Barber: Say no more.
    Args:
    max_param_norms :: numpy.ndarray - an array, shaped like
        the params at each axis0 position, that specifies the maximum
        absolute value of the parameters
    params :: numpy.ndarray - the parameters to be clipped
    Returns: none
    """
    for i in range(params.shape[1]):
        max_amp = max_param_norms[i]
        params[:,i] = np.clip(params[:,i], -max_amp, max_amp)


def clip_param_norms(max_param_norms, params):
    """
    Me: I need the entry-wise norms of the column entries of my
        parameter array to each be scaled to a fixed
        maximum norm if they exceed that norm
    Barber: u wot m8?
    Args:
    max_param_norms :: numpy.ndarray - an array, shaped like
        the params at each axis0 position, that specifies the maximum
        norm of those parameters
    params :: numpy.ndarray - the parameters to be clipped
    Returns: none
    """
    for i, max_param_norm in enumerate(max_param_norms):
        _params = params[:, i]
        mag_params = np.abs(_params)
        offending_indices = np.nonzero(np.less(max_param_norm, mag_params))
        offending_params = _params[offending_indices]
        resolved_params = (np.divide(offending_params, mag_params[offending_indices])
                           * max_param_norm)
        _params[offending_indices] = resolved_params
    #ENDFOR


def gen_params_cos(pulse_time, pulse_step_count, param_count,
                    max_param_norms, periods=10.):
    """
    Create a discrete, complex parameter set that is shaped like
    a cosine function.

    Args:
    pulse_time :: float - the duration of the pulse
    pulse_step_count :: int - the number of time steps at which
        parameters are discretized
    param_count :: int - how many parameters are at each time step
    max_param_norms :: numpy.ndarray - an array of shape
        (parameter_count) that,
        at each point, specifies the +/- value at which the parameter
        should be clipped
    periods :: float - the number of periods that the wave should complete

    Returns:
    params :: np.ndarray(pulse_step_count, param_count) - paramters for
        the specified pulse_step_count and param_count with a cosine fit
        that are complex by default
    """
    period = np.divide(pulse_step_count, periods)
    b = np.divide(2 * np.pi, period)
    params = np.zeros((pulse_step_count, param_count))
    
    # Create a wave for each parameter over all time
    # and add it to the parameters.
    for i in range(param_count):
        # Generate a cosine wave about y=0 with amplitude
        # half of the max.
        max_norm = max_param_norms[i]
        _params = (np.divide(max_norm, 2)
                   * np.cos(b * np.arange(pulse_step_count)))
        # Replace all parameters that have zero value
        # with small values.
        small_norm = max_norm * 1e-1
        _params = np.where(_params, _params, small_norm)
        params[:, i] = _params
    #ENDFOR

    # Mimic the cosine fit for the imaginary parts and normalize.
    params = (params - 1j * params) / np.sqrt(2)

    return params


NORM_TOLERANCE = 1e-10
def initialize_params(initial_params, max_param_norms,
                       pulse_time,
                       pulse_step_count, param_count):
    """
    Sanitize the initial_params and max_param_norms.
    Generate both if either was not specified.
    Args:
    initial_params :: numpy.ndarray - the user specified initial parameters
    max_param_norms :: numpy.ndarray - the user specified max
        param amplitudes
    pulse_time :: float - the duration of the pulse
    pulse_step_count :: int - number of pulse steps
    param_count :: int - number of parameters per pulse step

    Returns:
    params :: numpy.ndarray - the initial parameters
    max_param_norms :: numpy.ndarray - the maximum parameter
        amplitudes
    """
    if max_param_norms is None:
        max_param_norms = np.ones(param_count)
        
    if initial_params is None:
        params = gen_params_cos(pulse_time, pulse_step_count, param_count,
                                max_param_norms)
    else:
        # If the user specified initial params, check that they conform to
        # max param amplitudes.
        for i, step_params in enumerate(initial_params):
            if not (np.less_equal(np.abs(step_params), max_param_norms + NORM_TOLERANCE).all()):
                raise ValueError("Expected that initial_params specified by "
                                 "user conformed to max_param_norms, but "
                                 "found conflict at step {} with {} and {}"
                                 "".format(i, step_params, max_param_norms))
        #ENDFOR
        params = initial_params

    return params, max_param_norms


def slap_params(gstate, params):
    """
    Reshape and transform parameters displayed to the optimizer
    to parameters understood by the cost function.
    Args:
    gstate :: qoc.GrapeState - information about the optimization
    params :: numpy.ndarray - the params in question
    Returns:
    new_params :: numpy.ndarray - the reshapen, transformed params
    """
    # Transform the parameters to C if they are complex.
    if gstate.complex_params:
        params = real_imag_to_complex_flat(params)
    # Reshape the parameters.
    params = np.reshape(params, gstate.params_shape)
    # Clip the parameters.
    clip_param_norms(gstate.max_param_norms, params)
    
    return params


def strip_params(gstate, params):
    """
    Reshape and transform parameters understood by the cost
    function to parameters understood by the optimizer.
    gstate :: qoc.GrapeState - information about the optimization
    params :: numpy.ndarray - the params in question
    Returns:
    new_params :: numpy.ndarray - the reshapen, transformed params
    """
    # Flatten the parameters.
    params = params.flatten()
    # Transform the parameters to R2 if they are complex.
    if gstate.complex_params:
        params = complex_to_real_imag_flat(params)

    return params


### MODULE TESTS ###

_BIG = 100

def _test():
    """
    Run test on the module's methods.
    """
    # Test parameter optimizer transformations.
    gstate = Dummy()
    gstate.complex_params = True
    shape_range = np.arange(_BIG) + 1
    for step_count in shape_range:
        for param_count in shape_range:
            gstate.params_shape = params_shape = (step_count, param_count)
            gstate.max_param_norms = np.ones(param_count) * 2
            params = np.random.rand(*params_shape) + 1j * np.random.rand(*params_shape)
            stripped_params = strip_params(gstate, params)
            assert(stripped_params.ndim == 1)
            assert(not (stripped_params.dtype in (np.complex64, np.complex128)))
            transformed_params = slap_params(gstate, stripped_params)
            assert(np.allclose(params, transformed_params))
            assert(params.shape == transformed_params.shape)
    #ENDFOR

    gstate.complex_params = False
    for step_count in shape_range:
        for param_count in shape_range:
            gstate.params_shape = params_shape = (step_count, param_count)
            gstate.max_param_norms = np.ones(param_count)
            params = np.random.rand(*params_shape)
            stripped_params = strip_params(gstate, params)
            assert(stripped_params.ndim == 1)
            assert(not (stripped_params.dtype in (np.complex64, np.complex128)))
            transformed_params = slap_params(gstate, stripped_params)
            assert(np.allclose(params, transformed_params))
            assert(params.shape == transformed_params.shape)
    #ENDFOR

    # Test parameter clipping.
    for step_count in shape_range:
        for param_count in shape_range:
            params_shape = (step_count, param_count)
            max_param_norms = np.ones(param_count)
            params = np.random.rand(*params_shape) * 2
            clip_params(max_param_norms, params)
            for step_params in params:
                assert(np.less_equal(step_params, max_param_norms).all())
            params = np.random.rand(*params_shape) * -2
            clip_params(max_param_norms, params)
            for step_params in params:
                assert(np.less_equal(-max_param_norms, step_params).all())
        #ENDFOR
    #ENDFOR

    # Parameter norm clipping.
    params = np.array(((1+2j, 7+8j), (3+4j, 5), (5+6j, 10,), (1-3j, -10),))
    max_param_norms = np.array((7, 8,))
    expected_clipped_params = np.array(((1+2j, (7+8j) * np.divide(8, np.sqrt(113))),
                                        (3+4j, 5),
                                        ((5+6j) * np.divide(7, np.sqrt(61)), 8,),
                                        (1-3j, -8)))
    clip_param_norms(max_param_norms, params)
    
    assert(np.allclose(params, expected_clipped_params))


if __name__ == "__main__":
    _test()
