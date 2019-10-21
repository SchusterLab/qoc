"""
bandpassfilter.py - This module defines a cost function that penalizes
certain frequency ranges of the control pulses.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class BandpassFilter(Cost):
    """
    This cost penalizes control pulses that contain forbidden frequencies.

    Fields:
    band_range_list :: ndarray 
        - A list of of lists of tuples that represents ranges of frequencies in GHz 
        that should be penalized
    name
    requires_step_evaluation
    """
    name = "bandpass_filter"
    requires_step_evaluation = False

    def __init__(self, band_range_list, control_count,
                 control_eval_count, evolution_time,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_count
        control_eval_count
        evolution_time
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.band_range_list = band_range_list
        self.control_count = control_count
        self.control_eval_times = anp.linspace(0, evolution_time, control_eval_count)
        self.dt = evolution_time / control_eval_count
    
    def cost(self, controls, states, system_eval_step):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        system_eval_step

        Returns:
        cost
        """
        # Find the frequency bins
        flist = np.fft.fftfreq(self.control_eval_times.shape[-1],d=self.dt)
        cost = 0
        #iterate over the controls (each control has a list of tuples of forbidden band ranges)
        for i, band_range in enumerate(self.band_range_list):
            control_fft_raw = anp.fft.fft(controls[:, i])
            control_fft_sq  = anp.real(anp.multiply(control_fft_raw, anp.conjugate(control_fft_raw)))
            #iterate over the forbidden tuples
            for j, band in enumerate(band_range):
                lower_bound_index = np.abs(flist-band[0]).argmin()
                upper_bound_index = np.abs(flist-band[1]).argmin()
                control_fft_sq_bound = control_fft_sq[lower_bound_index : upper_bound_index]
                cost += anp.sum(control_fft_sq_bound) / control_fft_sq_bound.shape[0]
        cost_normalized = cost / self.control_count
                       
        return cost_normalized * self.cost_multiplier
