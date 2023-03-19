"""
controlbandwidthmax.py - This module defines a cost function that penalizes all
control frequencies above a specified maximum.
"""

import autograd.numpy as anp
import numpy as np

from qoc.models import Cost

class ControlBandwidthMax(Cost):
    """
    This cost penalizes control frequencies above a set maximum.

    Fields:
    max_bandwidths :: ndarray (CONTROL_COUNT, 2) - This array contains the minimum and maximum allowed bandwidth of each control.
    control_count
    dt
    name
    requires_step_evaluation
    type
    
    Example Usage:
    dt = 1 # zns
    MAX_BANDWIDTH_0 = [0.01,0.4] # GHz
    MAX_BANDWIDTHS = anp.array[MAX_BANDWIDTH_0,] for single control field
    COSTS = [ControlBandwidthMax(dt, MAX_BANDWIDTHS)]
    """
    name = "control_bandwidth_max"
    requires_step_evaluation = False
    type = "control_explicit_related"
    def __init__(self, dt, max_bandwidths,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        dt
        max_bandwidths
        cost_multiplier
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_bandwidths = max_bandwidths
        self.dt = dt

        
    def cost(self, controls, states, gradients_method):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        gradients_method

        Returns:
        cost
        """
        control_eval_count = len(controls[0])
        freqs = np.fft.fftfreq(control_eval_count, d=self.dt)
        cost = 0
        # Iterate over the controls, penalize each control that has
        # frequencies greater than its maximum frequency or smaller than its minimum frequency.
        for i, bandwidth in enumerate(self.bandwidths):
            min_bandwidths = bandwidth[0]
            max_bandwidths = bandwidth[1]
            control_fft = anp.fft.fft(controls[i])
            control_fft_sq = anp.abs(control_fft)
            penalty_freq_indices_max = anp.nonzero(anp.abs(freqs) >= max_bandwidths)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices_max]
            penalty = anp.sum(penalized_ffts)
            penalty_freq_indices_min = anp.nonzero(anp.abs(freqs) <= min_bandwidths)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices_min]
            penalty = penalty + anp.sum(penalized_ffts)
            cost = cost + penalty
        return cost * self.cost_multiplier
