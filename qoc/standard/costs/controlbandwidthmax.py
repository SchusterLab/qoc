"""
controlbandwidthmax.py - This module defines a cost function that penalizes all
control frequencies above a specified maximum.
"""


import numpy as np

from qoc.models import Cost
import autograd.numpy as anp
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
    MAX_BANDWIDTHS = np.array((MAX_BANDWIDTH_0,))
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

    def cost(self, controls, states, system_eval_step,manual_mode=None):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        system_eval_step

        Returns:
        cost
        """
        if manual_mode==True:
            cost = 0
            self.penalty_freq_indices=[]
            self.normalization=[]
            self.max_indices=[]
        # Iterate over the controls, penalize each control that has
        # frequencies greater than its maximum frequency.
            for i, max_bandwidth in enumerate(self.max_bandwidths):
                control_fft = np.fft.fft(controls[:, i])
                control_fft_sq  = np.abs(control_fft)
                penalty_freq_indices = np.nonzero(self.freqs >= max_bandwidth)[0]
                self.penalty_freq_indices.append(penalty_freq_indices)
                penalized_ffts = control_fft_sq[penalty_freq_indices]
                penalty = np.sum(penalized_ffts)
                penalty_normalized = penalty / (penalty_freq_indices.shape[0] * np.max(penalized_ffts))
                self.normalization.append( penalty_freq_indices.shape[0])
                index=penalty_freq_indices[(np.max(penalized_ffts).argmax())]
                self.max_indices.append(index)
                cost = cost + penalty_normalized
            cost_normalized =  cost / self.control_count
        else:
            cost = 0
            # Iterate over the controls, penalize each control that has
            # frequencies greater than its maximum frequency.
            for i, max_bandwidth in enumerate(self.max_bandwidths):
                control_fft = anp.fft.fft(controls[:, i])
                control_fft_sq = anp.abs(control_fft)
                penalty_freq_indices = anp.nonzero(self.freqs >= max_bandwidth)[0]
                penalized_ffts = control_fft_sq[penalty_freq_indices]
                penalty = anp.sum(penalized_ffts)
                penalty_normalized = penalty / (penalty_freq_indices.shape[0] * anp.max(penalized_ffts))
                cost = cost + penalty_normalized
            cost_normalized = cost / self.control_count

        return cost_normalized * self.cost_multiplier

    def gradient_initialize(self, reporter):
        return
    def update_state(self, propagator):
        return

    def gradient(self,controls,j,k):
        grads_=0
        N = len(controls)
        maximum_index=self.max_indices[k]
        max=0
        for i in range(len(controls)):
            max = max + controls[i][k] * np.exp(-1j * 2 * i * maximum_index * np.pi / N)
        for m in range(len(self.penalty_freq_indices[k])):
            fre_number=self.penalty_freq_indices[k][m]
            fft=0
            for i in range(len(controls)):
                fft=fft+controls[i][k]*np.exp(-1j*2*i*fre_number*np.pi/N)
            current_grad=(np.real(fft) * np.real(np.exp(-1j * 2 * j * fre_number * np.pi / N)) + np.imag(fft) * np.imag(
                np.exp(-1j * 2 * j * fre_number * np.pi / N))) / np.abs(fft)
            max_grad=(np.real(max) * np.real(np.exp(-1j * 2 * j * maximum_index * np.pi / N)) + np.imag(max) * np.imag(
                np.exp(-1j * 2 * j * maximum_index * np.pi / N))) / np.abs(max)
            grads=(current_grad*np.abs(max)-max_grad*np.abs(fft))/(np.abs(max)**2)
            grads=grads/self.normalization[k]
            grads_=grads_+grads
        grads_=grads_/ self.control_count
        return  grads_

