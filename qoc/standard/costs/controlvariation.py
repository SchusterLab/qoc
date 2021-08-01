"""
controlvariation.py - This module defines a cost function
that penalizes variations of the control parameters.
"""


import numpy as np

from qoc.models import Cost
import autograd.numpy as anp
class ControlVariation(Cost):
    """
    This cost penalizes the variations of the control parameters
    from one `control_eval_step` to the next.

    Fields:
    control_size
    cost_multiplier
    cost_normalization_constant
    max_control_norms
    name
    order
    requires_step_evaluation
    """
    name = "control_variation"
    requires_step_evaluation = False

    def __init__(self, control_count,
                 control_eval_count,
                 cost_multiplier=1.,
                 max_control_norms=None,
                 order=1):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_count
        control_eval_count
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_control_norms = max_control_norms
        self.diffs_size = control_count * (control_eval_count - order)
        self.order = order
        self.cost_normalization_constant = self.diffs_size * (2 ** self.order)
        self.type = "control"
        self.control_eval_account=control_eval_count

    def cost(self, controls, states, system_eval_step,manual_mode):
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
            if self.max_control_norms is not None:
                normalized_controls = controls / self.max_control_norms
            else:
                normalized_controls = controls

        # Penalize the square of the absolute value of the difference
        # in value of the control parameters from one step to the next.
            diffs = np.diff(normalized_controls, axis=0, n=self.order)
            cost = np.sum(np.real(diffs * np.conjugate(diffs)))
        # You can prove that the square of the complex modulus of the difference
        # between two complex values is l.t.e. 2 if the complex modulus
        # of the two complex values is l.t.e. 1 respectively using the
        # triangle inequality. This fact generalizes for higher order differences.
        # Therefore, a factor of 2 should be used to normalize the diffs.
            cost_normalized = cost / self.cost_normalization_constant
        else:
            if self.max_control_norms is not None:
                normalized_controls = controls / self.max_control_norms
            else:
                normalized_controls = controls

            # Penalize the square of the absolute value of the difference
            # in value of the control parameters from one step to the next.
            diffs = anp.diff(normalized_controls, axis=0, n=self.order)
            cost = anp.sum(anp.real(diffs * anp.conjugate(diffs)))
            # You can prove that the square of the complex modulus of the difference
            # between two complex values is l.t.e. 2 if the complex modulus
            # of the two complex values is l.t.e. 1 respectively using the
            # triangle inequality. This fact generalizes for higher order differences.
            # Therefore, a factor of 2 should be used to normalize the diffs.
            cost_normalized = cost / self.cost_normalization_constant
        return cost_normalized * self.cost_multiplier

    def gradient_initialize(self, reporter):
        return
    def update_state(self, propagator):
        return

    def gradient(self, controls,j,k):
        grads=self.cost_multiplier/self.cost_normalization_constant
        if self.max_control_norms is not None:
            grads = grads / (self.max_control_norms) ** 2
        if self.order==1:
                if j==0 :
                    grads=-grads*2*(controls[j+1][k]-controls[j][k])
                elif j == self.control_eval_account-1:
                    grads = grads * 2 * (controls[j ][k] - controls[j-1][k])
                else:
                    grads = grads * 2 *( (controls[j][k] - controls[j - 1][k])-(controls[j+1][k]-controls[j][k]))
        if self.order==2:
                if j==0 :
                    grads=grads*2*(controls[j+2][k]-2*controls[j+1][k]+controls[j][k])
                elif j==1:
                    grads = grads *(-4*(controls[j+1][k]-2*controls[j][k]+controls[j-1][k])
                                        +2*(controls[j+2][k]-2*controls[j+1][k]+controls[j][k]))
                elif j==self.control_eval_account-2:
                    grads=grads*(2*(controls[j][k]-2*controls[j-1][k]+controls[j-2][k])
                                    -4*(controls[j+1][k]-2*controls[j][k]+controls[j-1][k]))
                elif j == self.control_eval_account-1:
                    grads = grads * 2*(controls[j][k]-2*controls[j-1][k]+controls[j-2][k])
                else:
                    grads = grads *(2*(controls[j][k]-2*controls[j-1][k]+controls[j-2][k])
                                    -4*(controls[j+1][k]-2*controls[j][k]+controls[j-1][k])
                                        +2*(controls[j+2][k]-2*controls[j+1][k]+controls[j][k]))
        return grads
