from qoc.state_transfer import simulation
import numpy as np
pre=2*np.pi
dim=800
result = simulation(1, dim, 6, 6 * pre, 3 * pre, -0.225 * pre, 0.1 * pre, 0.25,1,"AG")