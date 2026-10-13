"""Shared helpers for the Monte Carlo plotting scripts.

Editable figures
----------------
`save_figure` writes, next to the usual images, a `.mplfig` file: the whole
matplotlib Figure (data included) serialized with pickle, the equivalent of a
MATLAB .fig. Reopen it with

    python open_fig.py path/to/figure.mplfig        # interactive window

or from Python / a notebook

    from mcplot import load_figure
    fig = load_figure("path/to/figure.mplfig")
    fig.axes[0].lines[0].set_linewidth(3)
    fig.savefig("new.pdf")

Pickles are tied to the matplotlib version: reopen them inside the MCPlot
environment (environment.yaml pins the matplotlib minor version).
"""
import glob
import os
import pickle
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from matplotlib import rcParams

FIG_EXT = ".mplfig"
E_CHARGE = 1.602176634e-19  # C


def set_style(font_size=20):
    """Serif fonts (Nimbus Roman, falling back to DejaVu Serif) of the given size."""
    rcParams.update({
        "font.size": font_size,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "DejaVu Serif"],
    })


# ----------------------------------------------------------------------------
# Figure I/O
# ----------------------------------------------------------------------------
def save_figure(fig, path, formats=("png",), dpi=200, editable=True, **savefig_kw):
    """Save `fig` as `<path>.<fmt>` for each format and, if `editable`, as `<path>.mplfig`.

    `path` must not contain the extension (dots in the name are kept as they are).
    Extra keyword arguments (e.g. bbox_inches="tight") go to `Figure.savefig`.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi, **savefig_kw)
    if editable:
        with open(path + FIG_EXT, "wb") as f:
            pickle.dump(fig, f)


def load_figure(path):
    """Load a figure saved by `save_figure` (path with or without the .mplfig extension)."""
    if not path.endswith(FIG_EXT):
        path += FIG_EXT
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Simulation output readers
# ----------------------------------------------------------------------------
def read_subhistory(file):
    """Read a subhistory_<i>.txt file and return (electrons, holes) data rows.

    The first line holds the number of electrons and holes in fields 1 and 2,
    followed by one row per particle (electrons first, then holes).
    Columns used by the scripts: 3 = energy (eV), 7:10 = k (1/m), 10:13 = v (m/s).
    """
    with open(file) as f:
        sizes = f.readline().split()
    n_e, n_h = int(sizes[1]), int(sizes[2])
    data = np.loadtxt(file, skiprows=1, ndmin=2)
    return data[:n_e], data[n_e:n_e + n_h]


def read_current_voltage(file_path, contact_name):
    """Read contact_current.csv and return (current, voltage, time, std_dev) Series."""
    df = pd.read_csv(file_path)
    current = df[f"{contact_name} Total [A]"]
    voltage = df[f"{contact_name} Total [V]"]
    time_s = df["Time[s]"]
    std_dev = df[f"stddev({contact_name}) Total [A]"]
    return current, voltage, time_s, std_dev


def extract_vdc_from_folder(folder_name):
    """Extract the bias from a folder name of the form "<path>-<v>V".

    Examples:
        MC/1SC/voltage_ramp--37.00V -> -37.0
        FF/breakdown-15.00V         -> 15.0
        voltage_ramp--30.50V-s3     -> -30.5
    """
    match = re.search(r"-(-?\d+\.\d+)V", folder_name)
    return float(match.group(1)) if match else None


def read_ionization_times(file_path):
    """Return the impact ionization event times (s), first column of ionization.csv, or None if missing."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    times = pd.read_csv(file_path, header=None, usecols=[0]).squeeze("columns")
    return pd.to_numeric(times, errors="coerce").dropna().to_numpy()


def read_charge_collection(file_path, contact_name, dt):
    """Collected charge and the current obtained by differentiating it over intervals of `dt`.

    Returns (charge, time, currents, times): `currents` are the average currents over
    consecutive intervals of length dt, `times` the interval mid-points.
    """
    df = pd.read_csv(file_path)
    charge = df[f"{contact_name} ac_carriers [e]"].to_numpy()
    time_s = df["Time[s]"]
    t = time_s.to_numpy()

    currents, times = [], []
    for i in range(int((t[-1] - t[0]) / dt)):
        t1 = t[0] + i * dt
        idx1 = np.abs(t - t1).argmin()
        idx2 = np.abs(t - (t1 + dt)).argmin()
        if idx1 != idx2:
            currents.append((charge[idx2] - charge[idx1]) * E_CHARGE / (t[idx2] - t[idx1]))
            times.append(t1 + dt / 2)
    return charge, time_s, currents, times


def seed_number(path):
    """Seed of a folder named "...-s<seed>" (0 if absent)."""
    match = re.search(r"-s(\d+)$", path)
    return int(match.group(1)) if match else 0


def find_seed_folders(base_path, prefixes, voltages, label_fn=None):
    """Find the seed folders <base_path>/<prefix>-<V>V-s<seed> for every prefix.

    Returns {label: [folders relative to base_path, sorted by seed]}. The label is
    `label_fn(prefix)`, by default the second to last path component of the prefix.
    """
    if label_fn is None:
        label_fn = lambda p: p.split("/")[-2] if "/" in p else p
    folders_by_prefix = {}
    for prefix in prefixes:
        matches = [f for v in voltages for f in glob.glob(f"{base_path}/{prefix}-{v:.2f}V-s*")]
        folder_list = [os.path.relpath(f, base_path) for f in sorted(matches, key=seed_number)]
        label = label_fn(prefix)
        folders_by_prefix[label] = folder_list

        print(f"\n{label}: Found {len(folder_list)} folders")
        for f in folder_list[:5]:
            print(f"  {f}")
        if len(folder_list) > 5:
            print(f"  ... and {len(folder_list) - 5} more")
    return folders_by_prefix


# ----------------------------------------------------------------------------
# Parallel execution
# ----------------------------------------------------------------------------
def run_parallel(func, inputs, threads, what="plots"):
    """Map `func` over `inputs` with a process pool, printing the elapsed time."""
    print(f"Using {threads} processes for {what}. This may take a while...")
    start = time.time()
    with Pool(threads) as pool:
        results = pool.map(func, inputs)
    print(f"{what.capitalize()} created in {time.time() - start:.2f} seconds.")
    return results
