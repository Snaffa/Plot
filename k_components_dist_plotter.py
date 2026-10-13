"""Electron and hole wave-vector component distributions, one figure per subhistory.

Each figure has three panels (kx, ky, kz), in units of 2*pi/a.
"""
import os
from dataclasses import dataclass
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcplot import read_subhistory, run_parallel, save_figure

A_SI = 5.431e-10  # Si lattice constant (m)

# High-symmetry points, in units of 2*pi/a (adjust as needed for your lattice)
BZ_POINTS = {"Γ": 0.0, "X": 0.5, "L": 0.5, "W": 0.375, "K": 0.375, "Λ": 0.25, "Z": 0.5}


@dataclass
class Config:
    folder: str  # folder containing subhistory_<i>.txt
    save_folder_e: str
    save_folder_h: str
    prefix: str  # saved files: <prefix>_<id:04d>
    dt: float  # time per subhistory (s); the title shows t = dt*id
    num_count_bins: int = 100
    min_count_k: float = -1.0  # units of 2*pi/a
    max_count_k: float = 1.0
    bz_ticks: bool = False  # label the x axis with the Brillouin-zone high-symmetry points
    editable: bool = True  # also save the .mplfig editable figure


SPECIES = {"e": ("Electron", "b"), "h": ("Hole", "r")}


def add_bz_ticks(ax, xmin, xmax):
    """Put the high-symmetry point labels (±value) as x ticks within [xmin, xmax]."""
    ticks = sorted({(s * val, lab) for lab, val in BZ_POINTS.items() for s in (-1, 1)
                    if xmin <= s * val <= xmax})
    if ticks:
        ax.set_xticks([pos for pos, _ in ticks])
        ax.set_xticklabels([lab for _, lab in ticks], fontsize=8)


def plot_species(k_vectors, species, cfg, t, frame_id):
    name, color = SPECIES[species]
    edges = np.linspace(cfg.min_count_k, cfg.max_count_k, cfg.num_count_bins)
    dk = abs(edges[1] - edges[0])
    k_norm = k_vectors / (2 * np.pi / A_SI)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    for ax, comp, kc in zip(axes, "xyz", k_norm.T):
        ax.hist(kc, bins=edges, color=color, edgecolor="k", alpha=0.7, density=True,
                label=f"$k_{comp}$, dk = {dk:.3f} 2π/a")
        ax.set_title(f"Normalized {name} k Distribution, {comp} component t = {t:.2e}s")
        ax.set_xlim([cfg.min_count_k, cfg.max_count_k])
        ax.set_ylabel("Counts")
        ax.set_xlabel(f"$k_{comp}$ (2π/a)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if cfg.bz_ticks:
            add_bz_ticks(ax, cfg.min_count_k, cfg.max_count_k)

    fig.tight_layout()
    save_folder = cfg.save_folder_e if species == "e" else cfg.save_folder_h
    save_figure(fig, os.path.join(save_folder, f"{cfg.prefix}_{frame_id:04d}"),
                dpi=200, editable=cfg.editable)
    plt.close(fig)


def plot_frame(cfg, frame_id):
    electrons, holes = read_subhistory(os.path.join(cfg.folder, f"subhistory_{frame_id}.txt"))
    t = cfg.dt * frame_id
    for species, particles in (("e", electrons), ("h", holes)):
        if len(particles) > 0:
            plot_species(particles[:, 7:10], species, cfg, t, frame_id)


def main():
    threads = 32
    total_sh = 11  # number of subhistories

    base_dir = "P:/mnt/polito/MonteCarlo/DEMC/"
    device_dir = "pn/"
    simulation_dir = "bias-0V-1fs"

    cfg = Config(
        folder=os.path.join(base_dir, device_dir, simulation_dir),
        save_folder_e=os.path.join(device_dir, simulation_dir, "k_frames_e"),
        save_folder_h=os.path.join(device_dir, simulation_dir, "k_frames_h"),
        prefix="k_dist",
        dt=1e-12,
        num_count_bins=200,
        min_count_k=-1,
        max_count_k=1,
    )
    os.makedirs(cfg.save_folder_e, exist_ok=True)
    os.makedirs(cfg.save_folder_h, exist_ok=True)

    run_parallel(partial(plot_frame, cfg), range(total_sh), threads, "k distribution plots")


if __name__ == "__main__":
    main()
