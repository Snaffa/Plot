"""Electron and hole energy distributions, one figure per subhistory.

Each figure has two panels: counts histogram (log scale) and the peak-normalized
distribution compared with a Maxwell energy distribution at temperature T.
"""
import os
from dataclasses import dataclass
from functools import partial

import matplotlib
matplotlib.use("Agg")  # batch rendering in worker processes
import matplotlib.pyplot as plt
import numpy as np

from mcplot import read_subhistory, run_parallel, save_figure

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K
HBAR_OMEGA_OP = 0.063  # optical phonon energy in Si, eV


@dataclass
class Config:
    folder: str  # folder containing subhistory_<i>.txt
    save_folder_e: str
    save_folder_h: str
    prefix: str  # saved files: <prefix>_<id:04d>
    dt: float  # time per subhistory (s); the title shows t = dt*id
    num_pdf_bins: int = 100
    max_pdf_ev: float = 6.0
    num_count_bins: int = 100
    max_count_ev: float = 6.0
    max_count: float = 1e6  # y limit of the counts panel
    remove_phonon_peaks: bool = False  # zero the bins at multiples of the optical phonon energy
    T: float = 300.0  # temperature of the reference Maxwell distribution (K)
    editable: bool = True  # also save the .mplfig editable figure


# name, histogram colour, Maxwell line style
SPECIES = {
    "e": ("Electron", "b", "r--"),
    "h": ("Hole", "r", "b--"),
}


def maxwell_energy(E, T):
    """3D Maxwell energy distribution at temperature T, normalized to peak = 1."""
    kT = KB_EV * T
    f = (2.0 / np.sqrt(np.pi)) * np.sqrt(E) / kT**1.5 * np.exp(-E / kT)
    return f / f.max() if f.max() > 0 else f


def plot_species(energy, species, cfg, t, frame_id):
    name, color, maxwell_style = SPECIES[species]
    count_edges = np.linspace(0, cfg.max_count_ev, cfg.num_count_bins)
    pdf_edges = np.linspace(0, cfg.max_pdf_ev, cfg.num_pdf_bins)
    centers = 0.5 * (pdf_edges[:-1] + pdf_edges[1:])
    dE_count = count_edges[1] - count_edges[0]
    dE_pdf = pdf_edges[1] - pdf_edges[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    # Counts
    ax1.hist(energy, bins=count_edges, edgecolor="k", color=color, alpha=0.7,
             label=f"dE = {dE_count / 1e-3:.2f} meV")
    ax1.set_title(f"{name} Energy Distribution t = {t:.2e}s")
    ax1.set_ylim([0.8, cfg.max_count])
    ax1.set_yscale("log")
    ax1.set_ylabel("Counts")
    ax1.set_xlabel("Energy (eV)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Peak-normalized distribution
    counts, _ = np.histogram(energy, bins=pdf_edges)
    counts = counts.astype(float)
    if cfg.remove_phonon_peaks:
        tol = 1.5 * dE_pdf
        n_max = int(pdf_edges[-1] // HBAR_OMEGA_OP)
        peaks = np.any([np.abs(centers - n * HBAR_OMEGA_OP) < tol for n in range(1, n_max + 1)], axis=0)
        counts[peaks] = 0
    if counts.max() > 0:
        counts /= counts.max()
    ax2.bar(centers, counts, width=dE_pdf, color=color, edgecolor="k", alpha=0.7,
            label=f"dE = {dE_pdf / 1e-3:.2f} meV")
    ax2.plot(centers, maxwell_energy(centers, cfg.T), maxwell_style, label=f"Maxwell {cfg.T:g}K")
    ax2.set_title(f"Normalized {name} Energy Distribution t = {t:.2e}s")
    ax2.set_ylabel("Probability Density")
    ax2.set_xlabel("Energy (eV)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    save_folder = cfg.save_folder_e if species == "e" else cfg.save_folder_h
    suffix = "_noPOP" if cfg.remove_phonon_peaks else ""
    save_figure(fig, os.path.join(save_folder, f"{cfg.prefix}{suffix}_{frame_id:04d}"),
                dpi=200, editable=cfg.editable)
    plt.close(fig)


def plot_frame(cfg, frame_id):
    electrons, holes = read_subhistory(os.path.join(cfg.folder, f"subhistory_{frame_id}.txt"))
    t = cfg.dt * frame_id
    for species, particles in (("e", electrons), ("h", holes)):
        if len(particles) > 0:
            plot_species(particles[:, 3], species, cfg, t, frame_id)


def main():
    threads = 8
    total_sh = 11  # number of subhistories

    base_dir = "/mnt/csmhome2/amudano/MonteCarlo/DEMC"
    device_dir = "SPAD_FBK_v2/3D"
    simulation_dir = "5um/MC/1SC/try_1fs"

    cfg = Config(
        folder=os.path.join(base_dir, device_dir, simulation_dir),
        save_folder_e=os.path.join(device_dir, simulation_dir, "energy_frames_e"),
        save_folder_h=os.path.join(device_dir, simulation_dir, "energy_frames_h"),
        prefix="energy_dist",
        dt=30e-12,
        num_pdf_bins=200,
        max_pdf_ev=0.5,
        num_count_bins=200,
        max_count_ev=0.5,
        max_count=1e6,
        remove_phonon_peaks=False,
    )
    os.makedirs(cfg.save_folder_e, exist_ok=True)
    os.makedirs(cfg.save_folder_h, exist_ok=True)

    run_parallel(partial(plot_frame, cfg), range(total_sh), threads, "energy distribution plots")


if __name__ == "__main__":
    main()
