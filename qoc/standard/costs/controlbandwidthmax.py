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
    max_bandwidths :: ndarray (CONTROL_COUNT) - This array contains the maximum allowed bandwidth of each control.
    control_count
    freqs :: ndarray (CONTROL_EVAL_COUNT) - This array contains the frequencies of each of the controls.
    name
    requires_step_evaluation
    
    Example Usage:
    CONTROL_COUNT = 1
    EVOLUTION_TIME = 10 #ns
    CONTROL_EVAL_COUNT = 1000
    MAX_BANDWIDTH_0 = 0.4 # GHz
    MAX_BANDWIDTHS = anp.array((MAX_BANDWIDTH_0,))
    COSTS = [ControlBandwidthMax(CONTROL_COUNT, CONTROL_EVAL_COUNT,
                                 EVOLUTION_TIME, MAX_BANDWIDTHS)]
    """
    name = "control_bandwidth_max"
    requires_step_evaluation = False

    def __init__(self, control_count,
                 control_eval_count, evolution_time,
                 max_bandwidths,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_count
        control_eval_count
        evolution_time
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_bandwidths = max_bandwidths
        self.control_count = control_count
        dt = evolution_time / (control_eval_count - 1)
        self.control_eval_count=control_eval_count
        self.freqs = np.fft.fftfreq(control_eval_count, d=dt)
        self.type = "control"

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
        cost = 0
        # Iterate over the controls, penalize each control that has
        # frequencies greater than its maximum frequency.
        for i, max_bandwidth in enumerate(self.max_bandwidths):
            control_fft = anp.fft.fft(controls[:, i])
            control_fft_sq  = anp.abs(control_fft)
            penalty_freq_indices = anp.nonzero(self.freqs >= max_bandwidth)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices]
            penalty = anp.sum(penalized_ffts)
            penalty_normalized = penalty / (penalty_freq_indices.shape[0] * anp.max(penalized_ffts))
            cost = cost + penalty_normalized
        cost_normalized =  cost / self.control_count
        self.normalization=(penalty_freq_indices.shape[0] * anp.max(penalized_ffts))
        self.penalty_freq_indices=penalty_freq_indices
        return cost_normalized * self.cost_multiplier

    def gradient_initialize(self, reporter):
        return
    def update_state(self, propagator):
        return

    def gradient(self,controls,j,k):
        grads=self.cost_multiplier/self.normalization
        grads_=0
        cs=0
        sn=0
        para=[[],[]]
        for m in range(len(self.penalty_freq_indices)):
            for i in range(len(controls)):
                cs=cs+controls[self.penalty_freq_indices[m]][k]*np.cos(2*np.pi*i*self.penalty_freq_indices[m]/self.control_eval_count)
                sn=sn+controls[self.penalty_freq_indices[m]][k]*np.sin(2*np.pi*i*self.penalty_freq_indices[m]/self.control_eval_count)
            para[0].append(cs)
            para[1].append(sn)
        for i in range(len(self.penalty_freq_indices)):
            grads_=grads_+(grads*(np.cos(2*np.pi*j*self.penalty_freq_indices[i]/self.control_eval_count)*para[0][i]
                        +np.sin(2*np.pi*j*self.penalty_freq_indices[i]/self.control_eval_count)*para[1][i]))/np.sqrt(para[0][i]**2+para[1][i]**2)

        return grads_