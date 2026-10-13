"""Figures of current transients, impact ionization and spectra shared by the notebooks.

Every function returns the Figure, so it can be saved with mcplot.save_figure.
"""
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import savgol_filter

from mcplot import extract_vdc_from_folder, read_current_voltage, read_ionization_times

MU_A = r"Current $I_{D}$ ($\mu$A)"


def read_transient(base_path, folder, contact):
    """Read <base_path>/<folder>/contact_current.csv.

    Returns a namespace with t, current, voltage, std (numpy arrays) and vdc (bias from the folder name).
    """
    current, voltage, time_s, std = read_current_voltage(f"{base_path}/{folder}/contact_current.csv", contact)
    return SimpleNamespace(t=time_s.to_numpy(), current=current.to_numpy(dtype=float),
                           voltage=voltage.to_numpy(dtype=float), std=std.to_numpy(dtype=float),
                           vdc=extract_vdc_from_folder(folder))


def ionization_hist(ax, base_path, folder, time_edges, label):
    """Histogram (in ps) of the impact ionization events of a simulation; nothing if the file is missing."""
    times = read_ionization_times(f"{base_path}/{folder}/ionization.csv")
    if times is not None:
        ax.hist(times * 1e12, bins=time_edges * 1e12, edgecolor="k", alpha=0.4, label=label)


def current_and_voltage_figures(base_path, folders, labels, contact="ncontact", voltage_on_twin=False):
    """Current (log, uA), quenching voltage V_q and SPAD voltage drop V_d = Vbias - V_q vs time."""
    runs = [(read_transient(base_path, f, contact), lab) for f, lab in zip(folders, labels)]

    fig_i, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()
    for r, lab in runs:
        ax1.plot(r.t * 1e12, r.current / 1e-6, label=f"Vbias = {r.vdc} V, {lab} current")
        if voltage_on_twin:
            ax2.plot(r.t * 1e12, r.voltage, "r", label=f"Vbias = {r.vdc} V, {lab} voltage")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel(MU_A)
    ax1.set_yscale("log")
    ax1.set_title("SPAD current")
    ax1.grid(True)
    ax1.legend(loc="lower left")
    ax2.set_ylabel("Voltage (V)")
    if voltage_on_twin:
        ax2.legend(loc="upper right")
    fig_i.tight_layout()

    fig_vq, ax = plt.subplots(figsize=(8, 6))
    for r, lab in runs:
        ax.plot(r.t * 1e12, r.voltage, label=f"Vbias = {r.vdc} V, {lab}")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Quenching voltage, $V_q$")
    ax.grid(True)
    ax.legend()

    fig_vd, ax = plt.subplots(figsize=(8, 6))
    for r, lab in runs:
        if r.vdc is None:
            print(f"{lab}: no '-<V>V' bias in the folder name, V_d not plotted")
            continue
        ax.plot(r.t * 1e12, r.vdc - r.voltage, label=lab)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("SPAD voltage drop, $V_d$")
    ax.grid(True)
    ax.legend()
    return fig_i, fig_vq, fig_vd


def current_and_ionization_figure(base_path, folders, labels, contact="ncontact", figsize=(10, 7), n_bins=200):
    """Current transient (log) with the impact ionization events histogram on a twin y axis."""
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    for folder, lab in zip(folders, labels):
        r = read_transient(base_path, folder, contact)
        ax1.plot(r.t * 1e12, r.current, label=f"{lab} current")
        ionization_hist(ax2, base_path, folder, np.linspace(0, r.t.max(), n_bins),
                        f"{lab} impact ionization rate")
    ax2.set_ylabel("Impact ionization events")
    ax2.set_yscale("log")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Current (A)")
    ax1.set_yscale("log")
    ax1.set_title("Current transient and impact ionization events count")
    ax1.grid(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left").get_frame().set_alpha(0.9)
    return fig


def diode_and_capacitive_current_figure(base_path, folders, labels, contact="ncontact",
                                        capacitance=5e-15, legend_loc="upper left"):
    """|I_D| and the capacitive current |C dV/dt| (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for folder, lab in zip(folders, labels):
        r = read_transient(base_path, folder, contact)
        ax.plot(r.t * 1e12, np.abs(r.current), label="ID")
        ax.plot(r.t * 1e12, np.abs(capacitance * np.gradient(r.voltage) / np.gradient(r.t)), label=f"{lab}, IC")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Current (A)")
    ax.set_yscale("log")
    ax.set_title(f"Correnti, {os.path.basename(os.path.normpath(base_path))}")
    ax.grid(True)
    ax.legend(loc=legend_loc)
    return fig


def ionization_events_figure(base_path, folders, labels, t_stop, figsize=(15, 7), n_bins=200,
                             show_bin_width=True, reverse_legend=False):
    """Histograms of the impact ionization events in [0, t_stop] (log scale)."""
    time_edges = np.linspace(0, t_stop, n_bins)
    fig, ax = plt.subplots(figsize=figsize)
    for folder, lab in zip(folders, labels):
        if show_bin_width:
            lab = f"{lab}, dt = {time_edges[1] - time_edges[0]:.2e} s"
        ionization_hist(ax, base_path, folder, time_edges, lab)
    ax.set_ylabel("Impact ionization events")
    ax.set_xlabel("Time (ps)")
    ax.set_yscale("log")
    ax.grid(True)
    handles, labs = ax.get_legend_handles_labels()
    if reverse_legend:
        handles, labs = handles[::-1], labs[::-1]
    ax.legend(handles, labs)
    return fig


def iv_figure(base_path, bias, folder_fmt="voltage_ramp-{v:.2f}V", contact="pcontact",
              label="full-band-R2336", n_tail=10):
    """I-V characteristic: |mean of the last n_tail current samples| of every bias point."""
    current = [abs(read_transient(base_path, folder_fmt.format(v=v), contact).current[-n_tail:].mean())
               for v in bias]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bias, current, "o-", label=label)
    ax.set_title(f"IV characteristic SPAD {contact}")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.grid(True)
    ax.legend(loc="upper left")
    return fig


def current_transient_figure(base_path, folders, labels, contact, figsize=(10, 7), log=True,
                             legend_loc="upper right", sv_filter=None, label_fmt="{label}"):
    """Current transient in uA, optionally Savitzky-Golay filtered.

    sv_filter: None or (window_length, polyorder). label_fmt can use {label} and {vdc};
    pass legend_loc=None for no legend.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for folder, lab in zip(folders, labels):
        r = read_transient(base_path, folder, contact)
        current = savgol_filter(r.current, *sv_filter) if sv_filter else r.current
        ax.plot(r.t * 1e12, current / 1e-6, label=label_fmt.format(label=lab, vdc=r.vdc))
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(MU_A)
    title = f"Current transient SPAD {contact}"
    if sv_filter:
        title = f"Current transient diode {contact} - SV filter"
    ax.set_title(title)
    ax.grid(True)
    if legend_loc:
        ax.legend(loc=legend_loc)
    if log:
        ax.set_yscale("log")
    return fig


def current_fft_figure(base_path, folders, labels, contact="p-contact", sv_filter=(51, 3),
                       t_start=1.5e-12, t_stop=100e-12, dt=1e-15, ramo_counter=8):
    """Normalized spectrum of the filtered current in [t_start, t_stop) (valid only for constant dt).

    The current is sampled every dt*ramo_counter; the spectrum is shown in dB relative to its first
    non-zero frequency.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for folder, lab in zip(folders, labels):
        print(f"{base_path}/{folder}/contact_current.csv")
        r = read_transient(base_path, folder, contact)
        current = savgol_filter(r.current, *sv_filter)[(r.t >= t_start) & (r.t < t_stop)]
        spectrum = fft(current)
        f = fftfreq(len(spectrum), d=dt * ramo_counter)
        positive = f >= 0
        mag_db = 20 * np.log10(np.abs(spectrum[positive]))
        ax.plot(f[positive] * 1e-9, mag_db - mag_db[1], label=f"Vbias = {r.vdc} V, {lab}")
    ax.set_xlim(1, 510)
    ax.set_xlabel("Frequency, GHz")
    ax.set_ylabel("Current (dB)")
    ax.set_xscale("log")
    ax.set_title("SPAD current fft")
    ax.grid(True)
    ax.legend()
    return fig


def scattering_rate_figure(scattering_file, xlim=(0, 4)):
    """Scattering rate vs energy from a file whose first line is the number of data rows."""
    with open(scattering_file) as f:
        n_rows = int(f.readline().strip())
        data = np.loadtxt(f, max_rows=n_rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(data[:, 0], data[:, 1])
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Scattering Rate (1/s)")
    ax.set_title("Scattering Rate vs Energy")
    ax.set_xlim(xlim)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    return fig
