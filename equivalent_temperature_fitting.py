"""Equivalent temperature of electrons and holes vs time.

For each subhistory the peak-normalized energy distribution is fitted with a
Maxwellian A*sqrt(E)*exp(-E/kT); the fitted kT gives the equivalent temperature.
"""
import os
from dataclasses import dataclass
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from mcplot import read_subhistory, run_parallel, save_figure

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K


@dataclass
class Config:
    folder: str  # folder containing subhistory_<i>.txt
    num_pdf_bins: int = 100
    max_pdf_ev: float = 6.0


def normalized_maxwell_energy(E, A, kT):
    """A*sqrt(E)*exp(-E/kT), normalized to peak = 1."""
    E = np.clip(E, 1e-12, None)
    m = A * np.sqrt(E) * np.exp(-E / kT)
    return m / m.max() if m.max() > 0 else m


def fit_temperature(energy, edges):
    """Return the equivalent temperature (K) of `energy`, or NaN if the fit fails."""
    counts, _ = np.histogram(energy, bins=edges, density=True)
    if counts.max() > 0:
        counts = counts / counts.max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        (_, kT), _ = curve_fit(normalized_maxwell_energy, centers, counts,
                               p0=[1.0, KB_EV * 300.0], bounds=([0.0, 1e-8], [np.inf, np.inf]),
                               maxfev=5000)
    except Exception:
        return np.nan
    return kT / KB_EV


def frame_temperatures(cfg, frame_id):
    electrons, holes = read_subhistory(os.path.join(cfg.folder, f"subhistory_{frame_id}.txt"))
    edges = np.linspace(0, cfg.max_pdf_ev, cfg.num_pdf_bins)
    Te_e = fit_temperature(electrons[:, 3], edges) if len(electrons) > 0 else np.nan
    Te_h = fit_temperature(holes[:, 3], edges) if len(holes) > 0 else np.nan
    return Te_e, Te_h


def main():
    threads = 32
    dt = 1e-12  # time per subhistory (s)
    base_dir = "/mnt/csmhome2/amudano/MonteCarlo/DEMC"
    device_dir = "SPAD_FBK_v2/quasi_1D"
    prefix = "energy_dist"
    # (simulation folder, number of subhistories)
    simulations = [
        ("CONTINUE/quenching--37.00V", 101),
        ("CONTINUE/higher_doping/quenching--37.00V", 23),
    ]

    for simulation_dir, total_sh in simulations:
        cfg = Config(folder=os.path.join(base_dir, device_dir, simulation_dir),
                     num_pdf_bins=200, max_pdf_ev=2)
        save_folder = os.path.join(device_dir, simulation_dir, "equivalent_temperature")

        results = run_parallel(partial(frame_temperatures, cfg), range(total_sh), threads,
                               "temperature fits")
        Te_e, Te_h = np.array(results, dtype=float).T
        t = np.arange(total_sh) * dt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, Te_e, "o-", label="Electrons")
        ax.plot(t, Te_h, "s-", label="Holes")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Equivalent temperature (K)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(save_folder, f"{prefix}_Te_vs_time")
        save_figure(fig, out, dpi=200)
        plt.close(fig)
        print(f"Saved T_eq vs time to {out}.png")


if __name__ == "__main__":
    main()
