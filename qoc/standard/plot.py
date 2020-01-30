"""
plot.py - convenient visualization tools
"""

import ntpath
import os

from filelock import FileLock, Timeout
import h5py
import matplotlib
if not "DISPLAY" in os.environ:
    matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import linalg as la
from qutip import *
from IPython import display
import pandas as pd

from qoc.models import ProgramType
from qoc.standard.functions.convenience import conjugate_transpose

### CONSTANTS ###

COLOR_PALETTE = (
    "blue", "red", "green", "pink", "purple", "orange",
    "teal", "grey", "black", "cyan", "magenta", "brown",
    "azure", "beige", "coral", "crimson",
)
COLOR_PALETTE_LEN = len(COLOR_PALETTE)

### MAIN METHODS ###

def plot_controls(file_path, amplitude_unit="GHz", 
                  dpi=1000,
                  marker_style="o", save_file_path=None,
                  save_index=None,
                  show=False, time_unit="ns",
                  title=None,
                  fft_freq_lo=None,
                  fft_freq_hi=None,
                  fft_tick_count=None):
    """
    Plot the controls,  and their discrete fast fourier transform.

    Arguments:
    file_path

    amplitude_unit
    dpi
    marker_style
    save_file_path
    save_index
    show
    time_unit
    fft_freq_lo
    fft_freq_hi
    fft_tick_count

    Returns: None
    """
    # Open the file; extract data.
    file_lock_path = "{}.lock".format(file_path)
    try:
        with FileLock(file_lock_path):
            with h5py.File(file_path, "r") as file_:
                # If no save_index was specified, choose the save_index that achieved the lowest
                # error.
                if save_index is None:
                    save_index = np.argmin(file_["error"])
                complex_controls = file_["complex_controls"][()]
                controls = file_["controls"][save_index][()]
                evolution_time = file_["evolution_time"][()]
    except Timeout:
        print("Could not access specified file.")
        return
    #ENDWITH
    control_count = controls.shape[1]
    control_eval_count = controls.shape[0]
    control_eval_times = np.linspace(0, evolution_time, control_eval_count)
    control_dt = control_eval_times[1] - control_eval_times[0]
    controls_real = np.real(controls)
    controls_imag = np.imag(controls)
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    if title is None:
        title = file_name

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
    plt.suptitle(title)
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
    ax = plt.subplot(2, 1, 2)
    freq_axis = np.fft.fftfreq(control_eval_count, d=control_dt)
    if complex_controls:
        for i in range(control_count):
            i2 = i * 2
            color_fft_real = get_color(i2)
            color_fft_imag = get_color(i2 + 1)
            control_fft_real = np.real(np.fft.fft(controls[:, i]))
            control_fft_imag = np.imag(np.fft.fft(controls[:, i]))
            control_fft_squared_real = np.real(control_fft_real * np.conj(control_fft_real))
            control_fft_squared_imag = np.real(control_fft_imag * np.conj(control_fft_imag))
            ax.plot(freq_axis,
                    control_fft_squared_real, marker_style, color=color_fft_real,
                    ms=2,alpha=0.9)
            ax.plot(freq_axis,
                    control_fft_squared_imag, marker_style, color=color_fft_imag,
                    ms=2,alpha=0.9)
    #ENDFOR
    ax.set_xlabel("Frequency ({})".format(amplitude_unit))
    ax.set_ylabel("FFT")
    
    freq_index_max = np.argmax(freq_axis)
    if fft_freq_lo is None:
        fft_freq_lo = 0
    if fft_freq_hi is None:
        fft_freq_hi = freq_axis[freq_index_max]
    ax.set_xbound(lower=fft_freq_lo, upper=fft_freq_hi)
    
    if fft_tick_count is not None:
        ticks = np.linspace(fft_freq_lo, fft_freq_hi, fft_tick_count)
        ax.set_xticks(ticks)

    # Export.
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi)
        
    if show:
        plt.show()


def plot_density_population(file_path,
                            density_index=0,
                            dpi=1000,
                            marker_style="o",
                            save_file_path=None,
                            save_index=None,
                            show=False,
                            time_unit="ns",
                            titile=None):
    """
    Plot the evolution of the population levels for a density matrix.

    Arguments:
    file_path :: str - the full path to the H5 file
    
    density_index
    dpi
    marker_style
    save_file_path
    show
    state_index
    time_unit

    Returns: None
    """
    # Open file; extract data.
    file_lock_path = "{}.lock".format(file_path)
    try:
        with FileLock(file_lock_path):
            with h5py.File(file_path, "r") as file_:
                evolution_time = file_["evolution_time"][()]
                program_type = file_["program_type"][()]
                system_eval_count = file_["system_eval_count"][()]
                if program_type == ProgramType.EVOLVE.value:
                    intermediate_densities = file_["intermediate_densities"][:, density_index, :, :]
                else:
                    # If no save index was specified, choose the index that achieved
                    # the lowest error.
                    if save_index is None:
                        save_index = np.argmin(file_["error"])
                        intermediate_densities = file_["intermediate_densities"][save_index, :, density_index, :, :]
            #ENDWITH
        #ENDWITH
    except Timeout:
        print("Could not access the specified file.")
        return
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    if title is None:
        title = file_name
    hilbert_size = intermediate_densities.shape[-2]
    system_eval_times = np.linspace(0, evolution_time, system_eval_count)

    # Compile data.
    population_data = list()
    for i in range(hilbert_size):
        population_data_ = np.real(intermediate_densities[:, i, i])
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
    plt.suptitle(title)
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


def plot_state_population(file_path, 
                          dpi=1000,
                          marker_style="o",
                          save_file_path=None,
                          save_index=None,
                          show=False,
                          state_index=0,
                          time_unit="ns",
                          title=None):
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
    # Open file; extract data.
    file_lock_path = "{}.lock".format(file_path)
    try:
        with FileLock(file_lock_path):
            with h5py.File(file_path, "r") as file_:
                evolution_time = file_["evolution_time"][()]
                program_type = file_["program_type"][()]
                system_eval_count = file_["system_eval_count"][()]
                if program_type == ProgramType.EVOLVE.value:
                    intermediate_states = file_["intermediate_states"][:, state_index, :, :]
                else:
                    # If no save index was specified, choose the index that achieved
                    # the lowest error.
                    if save_index is None:
                        save_index = np.argmin(file_["error"])
                    intermediate_states = file_["intermediate_states"][save_index, :, state_index, :, :]
            #ENDWITH
        #ENDWITH
    except Timeout:
        print("Could not access the specified file.")
        return
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    if title is None:
        title = file_name
    hilbert_size = intermediate_states.shape[-2]
    system_eval_times = np.linspace(0, evolution_time, system_eval_count)

    # Compile data.
    intermediate_densities = np.matmul(intermediate_states, conjugate_transpose(intermediate_states))
    population_data = list()
    for i in range(hilbert_size):
        population_data_ = np.real(intermediate_densities[:, i, i])
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
    plt.suptitle(title)
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
