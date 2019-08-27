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

# constants
COLOR_PALETTE = [
    "blue", "red", "green", "pink", "purple", "orange",
    "teal", "grey", "black", "cyan", "magenta", "brown",
    "azure", "beige", "coral", "crimson",
]


def plot_best_trial(file_path, save_path=None,
                amplitude_unit="GHz", time_unit="ns",
                dpi=1000, marker_style="o"):
    """
    Plot information regarding the information of a the lowest total error
    trial of a grape optimization.
    Args:
    file_path :: str - the full path to the H5 file
    save_path :: str - the full path to save the png file to
    amplitude_unit :: str - the unit to display for the pulse amplitude
    time_unit :: str - the unit to display for the pulse duration
    dpi :: int - the quality of the image
    marker_style :: str - the style to plot as
    Returns: nothing
    """
    # Open file and extract data.
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    _file = h5py.File(file_path, "r")
    # TODO: track early termination so argmin doesn't find uninitialized
    # error values.
    errors = _file["error"]
    best_index = np.argmin(np.where(errors, errors, np.finfo(np.float64).max))
    controls = np.array(_file["controls"][best_index])
    controls_real = np.real(controls)
    controls_imag = np.imag(controls)
    control_count = _file["control_count"][()]
    control_step_count = _file["control_step_count"][()]
    evolution_time = _file["evolution_time"][()]
    time_per_step = np.divide(evolution_time, control_step_count)
    step_per_time = np.divide(control_step_count, evolution_time)

    # If the user did not specify a save path,
    # save the file to the current directory with
    # the data file's prefix.
    if save_path is None:
        save_path = "{}.png".format(file_name)

    # Create labels and extra content.
    patches = list()
    labels = list()
    for i in range(control_count):
        i2 = i * 2
        label_real = "control_{}_real".format(i)
        label_imag = "control_{}_imag".format(i)
        color_real = COLOR_PALETTE[i2]
        color_imag = COLOR_PALETTE[i2 + 1]
        labels.append(label_real)
        labels.append(label_imag)
        patches.append(mpatches.Patch(label=label_real, color=color_real))
        patches.append(mpatches.Patch(label=label_imag, color=color_imag))
    #ENDFOR

    # Plot the data.
    plt.figure()
    plt.suptitle(file_name)
    plt.figlegend(handles=patches, labels=labels, loc="upper right",
                  framealpha=0.5)
    # plt.subplots_adjust(hspace=0.8)
    subplot_count = 1

    # pulses
    plt.subplot(subplot_count, 1, 1)
    time_axis = time_per_step * np.arange(control_step_count)
    for i in range(control_count):
        i2 = i * 2
        color_real = COLOR_PALETTE[i2]
        color_imag = COLOR_PALETTE[i2 + 1]
        pulse_real = controls_real[:, i]
        pulse_imag = controls_imag[:, i]
        plt.plot(time_axis, pulse_real, marker_style,
                 color=color_real, ms=2, alpha=0.9)
        plt.plot(time_axis, pulse_imag, marker_style,
                 color=color_imag, ms=2, alpha=0.9)
    #ENDFOR
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Amplitude ({})".format(amplitude_unit))

    # # fft
    # plt.subplot(subplot_count, 1, 2)
    # freq_axis = np.arange(step_count) * np.divide(step_per_time, step_count)
    # for i, pulse_fft in enumerate(uks_fft):
    #     color = COLOR_PALETTE[i]
    #     plt.plot(freq_axis,
    #              pulse_fft, marker_style, color=color,
    #              ms=2,alpha=0.9)
    # #ENDFOR
    # plt.xlabel("Frequency ({})".format(amplitude_unit))
    # plt.ylabel("FFT")

    # # state population
    # for i in range(initial_vector_count):
    #     plt.subplot(subplot_count, 1, 3 + i)
    #     plt.xlabel("Time ({})".format(time_unit))
    #     plt.ylabel("Population")
    #     for j in range(hilbert_space_dimension):
    #         pop_data = population_data[i][j]
    #         color = COLOR_PALETTE[j]
    #         if i == 0:
    #             print("hs: {}, c: {}".format(j, color))
    #         plt.plot(time_axis, pop_data, marker_style, color=color, ms=2, alpha=0.5)
    #     #ENDFOR
    # #ENDFOR
    
    plt.savefig(save_path, dpi=dpi)
    

def _tests():
    """
    Run tests on the module.
    """
    pass


if __name__ == "__main__":
   _tests()
