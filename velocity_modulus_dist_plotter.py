"""Electron and hole velocity-modulus distributions, one figure per subhistory.

Each figure has two panels: the full peak-normalized distribution compared with
a Maxwell-Boltzmann speed distribution at temperature T, and a zoomed view.
"""
import os
from dataclasses import dataclass
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import electron_mass, k

from mcplot import read_subhistory, run_parallel, save_figure


@dataclass
class Config:
    folder: str  # folder containing subhistory_<i>.txt
    save_folder_e: str
    save_folder_h: str
    prefix: str  # saved files: <prefix>_<id:04d>
    dt: float  # time per subhistory (s); the title shows t = dt*id
    num_zoomed_bins: int = 100
    min_zoomed_v: float = 5e5  # m/s
    max_zoomed_v: float = 1e6  # m/s
    num_count_bins: int = 100
    min_count_v: float = 0.0  # m/s
    max_count_v: float = 1e6  # m/s
    T: float = 300.0  # temperature of the reference Maxwell distribution (K)
    editable: bool = True  # also save the .mplfig editable figure


# name, histogram colour, Maxwell line style, mass, y limit of the zoomed panel
SPECIES = {
    "e": ("Electron", "b", "r--", electron_mass, 0.1),
    "h": ("Hole", "r", "b--", 0.36 * electron_mass, 0.5),
}


def maxwell_speed(v, m, T):
    """Maxwell-Boltzmann speed distribution, normalized to peak = 1."""
    f = 4 * np.pi * (m / (2 * np.pi * k * T))**1.5 * v**2 * np.exp(-m * v**2 / (2 * k * T))
    return f / f.max()


def normalized_hist(ax, data, edges, color):
    """Histogram of `data` normalized to peak = 1."""
    counts, _ = np.histogram(data, bins=edges)
    weights = np.full(len(data), 1.0 / counts.max()) if counts.max() > 0 else None
    ax.hist(data, bins=edges, color=color, edgecolor="k", alpha=0.7, weights=weights,
            label=f"dv = {edges[1] - edges[0]:.2f} m/s")


def plot_species(speed, species, cfg, t, frame_id):
    name, color, maxwell_style, mass, zoom_ylim = SPECIES[species]
    edges = np.linspace(cfg.min_count_v, cfg.max_count_v, cfg.num_count_bins)
    zoomed_edges = np.linspace(cfg.min_zoomed_v, cfg.max_zoomed_v, cfg.num_zoomed_bins)
    v = np.linspace(cfg.min_count_v, cfg.max_count_v, 1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    normalized_hist(ax1, speed, edges, color)
    ax1.plot(v, maxwell_speed(v, mass, cfg.T), maxwell_style, label=f"Maxwell {cfg.T:g}K")
    ax1.set_title(f"Normalized {name} Velocity Distribution t = {t:.2e}s")
    ax1.set_xlim([cfg.min_count_v, cfg.max_count_v])

    normalized_hist(ax2, speed, zoomed_edges, color)
    ax2.set_title(f"Normalized {name} Velocity Distribution zoom t = {t:.2e}s")
    ax2.set_xlim([cfg.min_zoomed_v, cfg.max_zoomed_v])
    ax2.set_ylim([0, zoom_ylim])

    for ax in (ax1, ax2):
        ax.set_ylabel("Counts")
        ax.set_xlabel("Velocity (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()

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
            plot_species(np.linalg.norm(particles[:, 10:13], axis=1), species, cfg, t, frame_id)


def main():
    threads = 4
    dt = 2e-15  # time per subhistory (s)
    base_dir = "/mnt/csmhome2/amudano/MonteCarlo/DEMC"
    device_dir = "SPAD_FBK_v2/quasi_2D/half/bulk"
    # (simulation folder, number of subhistories)
    simulations = [
        ("CONTINUE/m30.5-1SC-fb-R500k-C5f-0.1fs", 25697),
        ("CONTINUE/m30.5-1SC-fb-largeRamo", 28341),
        ("CONTINUE/m30.5-1SC-fb-smallRamo", 29243),
    ]

    for simulation_dir, total_sh in simulations:
        cfg = Config(
            folder=os.path.join(base_dir, device_dir, simulation_dir),
            save_folder_e=os.path.join(device_dir, simulation_dir, "velocity_modulus_frames_e"),
            save_folder_h=os.path.join(device_dir, simulation_dir, "velocity_modulus_frames_h"),
            prefix="velocity_modulus_dist",
            dt=dt,
            num_zoomed_bins=100,
            min_zoomed_v=5e5,
            max_zoomed_v=1e6,
            num_count_bins=100,
            min_count_v=0,
            max_count_v=1e6,
            editable=False,  # ~0.5 MB per .mplfig: too much for tens of thousands of frames
        )
        os.makedirs(cfg.save_folder_e, exist_ok=True)
        os.makedirs(cfg.save_folder_h, exist_ok=True)

        run_parallel(partial(plot_frame, cfg), range(total_sh), threads,
                     "velocity distribution plots")


if __name__ == "__main__":
    main()
