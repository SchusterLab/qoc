"""
common.py - This module defines methods that are used by
multiple core functionalities.
"""

import numpy as np

from qoc.standard import(complex_to_real_imag_flat,
                         real_imag_to_complex_flat)

def clip_control_norms(max_control_norms, controls):
    """
    Me: I need the entry-wise norms of the column entries of my
        control array to each be scaled to a fixed
        maximum norm if they exceed that norm
    Barber: u wot m8?

    Args:
    max_control_norms :: ndarray (control_count) - an array that
        specifies the maximum norm for each control for all time
    controls :: ndarray - the controls to be clipped

    Returns: none
    """
    for i, max_control_norm in enumerate(max_control_norms):
        control = controls[:, i]
        mag_control = np.abs(control)
        offending_indices = np.nonzero(np.less(max_control_norm, mag_control))
        offending_control_points = control[offending_indices]
        resolved_control_points = (np.divide(offending_control_points, mag_control[offending_indices])
                                   * max_control_norm)
        control[offending_indices] = resolved_control_points
    #ENDFOR


def gen_controls_cos(complex_controls, control_count, control_step_count,
                     evolution_time,
                     max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a cosine function.

    Args:
    complex_controls :: bool - whether or not the controls should be complex
    control_count :: int - how many controls are given to the hamiltonian
        at each time step
    control_step_count :: int - the number of time steps at which
        controleters are discretized
    evolution_time :: float - the duration of the system evolution
    max_control_norms :: ndarray (control count) - an array that
        specifies the maximum norm for each control for all time
    periods :: float - the number of periods that the wave should complete

    Returns:
    controls :: ndarray(control_step_count, control_count) - controls for
        the specified control_step_count and control_count with a cosine fit
    """
    period = np.divide(control_step_count, periods)
    b = np.divide(2 * np.pi, period)
    controls = np.zeros((control_step_count, control_count))
    
    # Create a wave for each control over all time
    # and add it to the controls.
    for i in range(control_count):
        # Generate a cosine wave about y=0 with amplitude
        # half of the max.
        max_norm = max_control_norms[i]
        _controls = (np.divide(max_norm, 2)
                   * np.cos(b * np.arange(control_step_count)))
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


def gen_controls_flat(complex_controls, control_count, control_step_count,
                      evolution_time,
                      max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a flat line with small amplitude.
    """
    controls = np.zeros((control_step_count, control_count))

    # Make each control a flat line for all time.
    for i in range(control_count):
        max_norm = max_control_norms[i]
        small_norm = max_norm * 1e-1
        control = np.repeat(small_norm, control_step_count)
        controls[:, i] = control
    #ENDFOR

    # Mimic the flat line for the imaginary parts and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / np.sqrt(2)

    return controls


_NORM_TOLERANCE = 1e-10
def initialize_controls(complex_controls,
                        control_count,
                        control_step_count,
                        evolution_time,
                        initial_controls, max_control_norms,):
    """
    Sanitize `initial_controls` with `max_control_norms`.
    Generate both if either was not specified.

    Args:
    complex_controls :: bool - whether or not the controls should be complex
    control_count :: int - number of controls per control_step
    control_step_count :: int - number of pulse steps
    initial_controls :: ndarray (control_count, control_step_count)
        - the user specified initial controls
    max_control_norms :: ndarray (control_count) - the user specified max
        control norms
    evolution_time :: float - the duration of the pulse

    Returns:
    controls :: ndarray - the initial controls
    max_control_norms :: ndarray - the maximum control norms
    """
    if max_control_norms is None:
        max_control_norms = np.ones(control_count)
        
    if initial_controls is None:
        controls = gen_controls_flat(complex_controls, control_count, control_step_count,
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
                                 "max_control_norms={}"
                                 "".format(control_step, step_controls, max_control_norms))
        #ENDFOR
        controls = initial_controls

    return controls, max_control_norms


def slap_controls(gstate, controls):
    """
    Reshape and transform controleters displayed to the optimizer
    to controleters understood by the cost function.
    Args:
    gstate :: qoc.GrapeState - information about the optimization
    controls :: ndarray - the controls in question
    Returns:
    new_controls :: ndarray - the reshapen, transformed controls
    """
    # Transform the controleters to C if they are complex.
    if gstate.complex_controls:
        controls = real_imag_to_complex_flat(controls)
    # Reshape the controleters.
    controls = np.reshape(controls, gstate.controls_shape)
    # Clip the controleters.
    clip_control_norms(gstate.max_control_norms, controls)
    
    return controls


def strip_controls(gstate, controls):
    """
    Reshape and transform controleters understood by the cost
    function to controleters understood by the optimizer.
    gstate :: qoc.GrapeState - information about the optimization
    controls :: ndarray - the controls in question
    Returns:
    new_controls :: ndarray - the reshapen, transformed controls
    """
    # Flatten the controleters.
    controls = controls.flatten()
    # Transform the controleters to R2 if they are complex.
    if gstate.complex_controls:
        controls = complex_to_real_imag_flat(controls)

    return controls


### MODULE TESTS ###

_BIG = 100

def _test():
    """
    Run test on the module's methods.
    """
    from qoc.models.dummy import Dummy

    # Test control optimizer transformations.
    gstate = Dummy()
    gstate.complex_controls = True
    shape_range = np.arange(_BIG) + 1
    for step_count in shape_range:
        for control_count in shape_range:
            gstate.controls_shape = controls_shape = (step_count, control_count)
            gstate.max_control_norms = np.ones(control_count) * 2
            controls = np.random.rand(*controls_shape) + 1j * np.random.rand(*controls_shape)
            stripped_controls = strip_controls(gstate, controls)
            assert(stripped_controls.ndim == 1)
            assert(not (stripped_controls.dtype in (np.complex64, np.complex128)))
            transformed_controls = slap_controls(gstate, stripped_controls)
            assert(np.allclose(controls, transformed_controls))
            assert(controls.shape == transformed_controls.shape)
    #ENDFOR

    gstate.complex_controls = False
    for step_count in shape_range:
        for control_count in shape_range:
            gstate.controls_shape = controls_shape = (step_count, control_count)
            gstate.max_control_norms = np.ones(control_count)
            controls = np.random.rand(*controls_shape)
            stripped_controls = strip_controls(gstate, controls)
            assert(stripped_controls.ndim == 1)
            assert(not (stripped_controls.dtype in (np.complex64, np.complex128)))
            transformed_controls = slap_controls(gstate, stripped_controls)
            assert(np.allclose(controls, transformed_controls))
            assert(controls.shape == transformed_controls.shape)
    #ENDFOR

    # Test control clipping.
    for step_count in shape_range:
        for control_count in shape_range:
            controls_shape = (step_count, control_count)
            max_control_norms = np.ones(control_count)
            controls = np.random.rand(*controls_shape) * 2
            clip_controls(max_control_norms, controls)
            for step_controls in controls:
                assert(np.less_equal(step_controls, max_control_norms).all())
            controls = np.random.rand(*controls_shape) * -2
            clip_controls(max_control_norms, controls)
            for step_controls in controls:
                assert(np.less_equal(-max_control_norms, step_controls).all())
        #ENDFOR
    #ENDFOR

    # Control norm clipping.
    controls = np.array(((1+2j, 7+8j), (3+4j, 5), (5+6j, 10,), (1-3j, -10),))
    max_control_norms = np.array((7, 8,))
    expected_clipped_controls = np.array(((1+2j, (7+8j) * np.divide(8, np.sqrt(113))),
                                        (3+4j, 5),
                                        ((5+6j) * np.divide(7, np.sqrt(61)), 8,),
                                        (1-3j, -8)))
    clip_control_norms(max_control_norms, controls)
    
    assert(np.allclose(controls, expected_clipped_controls))


if __name__ == "__main__":
    _test()
