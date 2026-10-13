"""Electron and hole velocity-component distributions, one figure per subhistory.

Each figure has three panels (vx, vy, vz), each compared with the 1D Gaussian
(Maxwellian component) at temperature T for the given effective mass.
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
    num_count_bins: int = 100
    min_count_v: float = -1e6  # m/s
    max_count_v: float = 1e6  # m/s
    max_density: float = 4e-6  # y limit of the panels
    T: float = 300.0  # temperature of the reference Gaussian (K)
    editable: bool = True  # also save the .mplfig editable figure


# name, histogram colour, Gaussian line style, effective mass, Gaussian label
SPECIES = {
    "e": ("Electron", "b", "r--", 0.36 * electron_mass, "Gaussian {T:g}K ($v_{c}$)"),
    "h": ("Hole", "r", "b--", 0.49 * electron_mass, "Gaussian {T:g}K (hh)"),  # heavy hole
}


def gaussian_component(v, m, T):
    """Distribution of one velocity component: normal with <v^2> = kT/m."""
    sigma = np.sqrt(k * T / m)
    return np.exp(-v**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def plot_species(velocities, species, cfg, t, frame_id):
    name, color, gauss_style, mass, gauss_label = SPECIES[species]
    edges = np.linspace(cfg.min_count_v, cfg.max_count_v, cfg.num_count_bins)
    dv = abs(edges[1] - edges[0])
    v = np.linspace(cfg.min_count_v, cfg.max_count_v, 1000)
    gauss = gaussian_component(v, mass, cfg.T)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    for ax, comp, vel in zip(axes, "xyz", velocities.T):
        ax.hist(vel, bins=edges, color=color, edgecolor="k", alpha=0.7, density=True,
                label=f"$v_{comp}$, dv = {dv:.2f} m/s")
        ax.plot(v, gauss, gauss_style, label=gauss_label.format(T=cfg.T, c=comp))
        ax.set_title(f"Normalized {name} Velocity Distribution, {comp} component t = {t:.2e}s")
        ax.set_xlim([cfg.min_count_v, cfg.max_count_v])
        ax.set_ylim([0, cfg.max_density])
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
            plot_species(particles[:, 10:13], species, cfg, t, frame_id)


def main():
    threads = 8
    total_sh = 11  # number of subhistories

    base_dir = "/mnt/csmhome2/amudano/MonteCarlo/DEMC"
    device_dir = "SPAD_FBK_v2/3D"
    simulation_dir = "5um/MC/1SC/try_1fs"

    cfg = Config(
        folder=os.path.join(base_dir, device_dir, simulation_dir),
        save_folder_e=os.path.join(device_dir, simulation_dir, "velocity_frames_e"),
        save_folder_h=os.path.join(device_dir, simulation_dir, "velocity_frames_h"),
        prefix="velocity_dist",
        dt=30e-12,
        num_count_bins=200,
        min_count_v=-1e6,
        max_count_v=1e6,
    )
    os.makedirs(cfg.save_folder_e, exist_ok=True)
    os.makedirs(cfg.save_folder_h, exist_ok=True)

    run_parallel(partial(plot_frame, cfg), range(total_sh), threads, "velocity distribution plots")


if __name__ == "__main__":
    main()
