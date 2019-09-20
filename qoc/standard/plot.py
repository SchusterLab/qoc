"""
plot.py - convenient visualization tools
"""

import ntpath
import os

import h5py
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la

from qoc.standard.functions.convenience import conjugate_transpose

### CONSTANTS ###

COLOR_PALETTE = (
    "blue", "red", "green", "pink", "purple", "orange",
    "teal", "grey", "black", "cyan", "magenta", "brown",
    "azure", "beige", "coral", "crimson",
)
COLOR_PALETTE_LEN = len(COLOR_PALETTE)

### MAIN METHODS ###

def plot_best_controls(file_path, amplitude_unit="GHz", 
                       dpi=1000, marker_style="o", save_file_path=None,
                       show=False, time_unit="ns",):
    """
    Plot the controls,  and their discrete fast fourier transform, 
    that achieved the lowest error.

    Arguments:
    file_path

    amplitude_unit
    dpi
    marker_style
    save_file_path
    show
    time_unit

    Returns: None
    """
    # Open the file and extract data.
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    _file = h5py.File(file_path, "r")
    errors = _file["error"]
    best_index = np.argmin(errors)
    complex_controls = _file["complex_controls"][()]
    controls = _file["controls"][best_index][()]
    controls_real = np.real(controls)
    controls_imag = np.imag(controls)
    control_count = controls.shape[1]
    control_eval_count = controls.shape[0]
    evolution_time = _file["evolution_time"][()]
    control_eval_times = np.linspace(0, evolution_time, control_eval_count)

    # Create labels and extra content.
    patches = list()
    labels = list()
    for i in range(control_count):
        i2 = i * 2
        label_real = "control_{}_real".format(i)
        labels.append(label_real)
        color_real = get_color(i2)
        patches.append(mpatches.Patch(label=label_real, color=color_real))

        label_imag = "control_{}_imag".format(i)
        color_imag = get_color(i2 + 1)
        labels.append(label_imag)
        patches.append(mpatches.Patch(label=label_imag, color=color_imag))
    #ENDFOR

    # Set up the plots.
    plt.figure()
    plt.suptitle(file_name)
    plt.figlegend(handles=patches, labels=labels, loc="upper right",
                  framealpha=0.5)
    plt.subplots_adjust(hspace=0.8)

    # Plot the controls.
    plt.subplot(2, 1, 1)
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Amplitude ({})".format(amplitude_unit))
    if complex_controls:
        for i in range(control_count):
            i2 = i * 2
            color_real = get_color(i2)
            color_imag = get_color(i2 + 1)
            control_real = controls_real[:, i]
            control_imag = controls_imag[:, i]
            plt.plot(control_eval_times, control_real, marker_style,
                     color=color_real, ms=2, alpha=0.9)
            plt.plot(control_eval_times, control_imag, marker_style,
                     color=color_imag, ms=2, alpha=0.9)
    else:
        for i in range(control_count):
            i2 = i * 2
            color= get_color(i2)
            control = controls[:, i]
            plt.plot(control_eval_times, control, marker_style,
                     color=color, ms=2, alpha=0.9)
    #ENDIF

    # Plot the fft.
    plt.subplot(2, 1, 2)
    freq_axis = np.where(control_eval_times, control_eval_times, 1) ** -1
    for i in range(control_count):
        i2 = i * 2
        color_fft_real = get_color(i2)
        color_fft_imag = get_color(i2 + 1)
        control_fft = np.fft.fft(controls[:, i])
        control_fft_real = control_fft.real
        control_fft_imag = control_fft.imag
        plt.plot(freq_axis,
                 control_fft_real, marker_style, color=color_fft_real,
                 ms=2,alpha=0.9)
        plt.plot(freq_axis,
                 control_fft_imag, marker_style, color=color_fft_imag,
                 ms=2,alpha=0.9)
    #ENDFOR
    plt.xlabel("Frequency ({})".format(amplitude_unit))
    plt.ylabel("FFT")

    # Export.
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi)
        
    if show:
        plt.show()


def plot_population_states(file_path, 
                           dpi=1000,
                           marker_style="o",
                           save_file_path=None, show=False,
                           state_index=0,
                           time_unit="ns",):
    """
    Plot the evolution of the population levels for a state.

    Arguments:
    file_path :: str - the full path to the H5 file

    dpi
    marker_style
    save_file_path
    show
    state_index
    time_unit

    Returns: None
    """
    # Open file and extract data.
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    file_ = h5py.File(file_path, "r")
    evolution_time = file_["evolution_time"][()]
    system_eval_count = file_["system_eval_count"][()]
    states = file_["intermediate_states"][()]
    hilbert_size = states.shape[2]
    system_eval_times = np.linspace(0, evolution_time, system_eval_count)

    # Compile data.
    densities = np.matmul(states, conjugate_transpose(states))
    population_data = list()
    for i in range(hilbert_size):
        population_data_ = np.real(densities[:, state_index, i, i])
        population_data.append(population_data_)

    # Create labels and extra content.
    patches = list()
    labels = list()
    for i in range(hilbert_size):
        label = "{}".format(i)
        labels.append(label)
        color = get_color(i)
        patches.append(mpatches.Patch(label=label, color=color))
    #ENDFOR

    # Plot the data.
    plt.figure()
    plt.suptitle(file_name)
    plt.figlegend(handles=patches, labels=labels, loc="upper right",
                  framealpha=0.5)
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Population")
    for i in range(hilbert_size):
        color = get_color(i)
        plt.plot(system_eval_times, population_data[i], marker_style,
                 color=color, ms=2, alpha=0.9)

    # Export.
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi)

    if show:
        plt.show()


### HELPER FUNCTIONS ###

def get_color(index):
    """
    Retrive a color unique to `index`.

    Arguments:
    index

    Returns:
    color
    """
    return COLOR_PALETTE[index % COLOR_PALETTE_LEN]
