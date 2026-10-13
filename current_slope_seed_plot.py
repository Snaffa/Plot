"""Current growth rate (slope of log10 I vs t) vs bias, averaged over seeds.

For every prefix (device configuration) and bias, the seeds whose current
reaches I_avalanche are fitted with a line in log10(I) for I_min <= I <= I_max.
The gain (collected charge / q) is integrated up to the shortest simulated time
among all valid seeds, so that every seed is compared on the same interval.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from mcplot import (E_CHARGE, extract_vdc_from_folder, find_seed_folders, read_current_voltage,
                    save_figure, set_style)

set_style(20)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BASE_PATH = "/mnt/csmhome2/amudano/MonteCarlo/DEMC/SPAD_FBK_v2/quasi_1D"
VOLTAGES = np.linspace(-30, -40, 11)
# Folders are searched as <BASE_PATH>/<prefix>-<V>V-s<seed>
SUBFOLDER_PREFIXES = [
    "FF/lower_di_doping/si_1e17/breakdown",
    # "FF/lower_di_doping/breakdown",
    # "FF/lower_di_doping/si_1e18/breakdown",
]
CONTACT = "ncontact"

I_MIN = 1e-7  # minimum current for the fit (exclude initial transient/noise)
I_MAX = 5e-4  # maximum current for the fit
I_AVALANCHE = 1e-5  # minimum peak current to consider the avalanche developed

DEBUG_PLOTS = False  # per-voltage current transients with the fits
PLOT_GAIN = False  # gain vs bias
SAVE_DIR = "/home/angelo/scripts/plot/NUSOD2026/current_slope"


def load_seeds(base_path, folders_by_prefix):
    """Read every seed and classify it as valid (avalanche + enough fit points) or skipped.

    Returns ({label: {vdc: {'valid': [...], 'skipped': [...]}}}, global minimum t_max of valid seeds).
    """
    print("=" * 60)
    print("FIRST PASS: Finding global minimum tmax across all simulations")
    print("=" * 60)
    seeds = {}
    t_max_min = np.inf
    for label, folder_list in folders_by_prefix.items():
        print(f"\nScanning: {label}")
        seeds[label] = {}
        for folder in folder_list:
            vdc = extract_vdc_from_folder(folder)
            groups = seeds[label].setdefault(vdc, {"valid": [], "skipped": []})
            current_file = f"{base_path}/{folder}/contact_current.csv"
            try:
                current, _, time_s, _ = read_current_voltage(current_file, CONTACT)
            except Exception as e:
                print(f"    Error reading {current_file}: {e}")
                continue

            seed = {"t": time_s.to_numpy(), "current": np.abs(current.to_numpy())}
            if seed["current"].max() < I_AVALANCHE:
                groups["skipped"].append(seed)
                continue
            seed["mask_fit"] = (seed["current"] >= I_MIN) & (seed["current"] <= I_MAX)
            if seed["mask_fit"].sum() > 2:
                groups["valid"].append(seed)
                t_max_min = min(t_max_min, seed["t"][-1])

    print(f"\n>>> GLOBAL t_max_min for integration: {t_max_min * 1e12:.2f} ps <<<\n")
    return seeds, t_max_min


def analyze(seeds, t_max_min):
    """Fit slope and integrate gain for every valid seed; return per-prefix averages."""
    results = {}
    for label, by_vdc in seeds.items():
        print(f"\n{'=' * 60}\nProcessing: {label}\n{'=' * 60}")
        print(f"Found {len(by_vdc)} voltage(s): {list(by_vdc.keys())}")
        print(f"Fitting in current range: [{I_MIN:.0e}, {I_MAX:.0e}] A")
        print(f"Avalanche threshold: I_max >= {I_AVALANCHE:.0e} A")
        print(f"Integration time: {t_max_min * 1e12:.2f} ps (global)\n")

        res = {k: [] for k in ("vdc", "avg_slope", "std_slope", "avg_gain", "std_gain")}
        res["per_seed"] = {}
        for vdc in sorted(by_vdc):
            valid = by_vdc[vdc]["valid"]
            print(f"  Voltage {vdc}V: {len(valid)} valid, {len(by_vdc[vdc]['skipped'])} skipped")

            for seed in valid:
                t, current, mask = seed["t"], seed["current"], seed["mask_fit"]
                seed["slope"], seed["intercept"] = np.polyfit(t[mask], np.log10(current[mask]), 1)
                in_window = t <= t_max_min
                seed["gain"] = np.trapezoid(current[in_window], t[in_window]) / E_CHARGE

            if not valid:
                continue
            slopes = [s["slope"] for s in valid]
            gains = [s["gain"] for s in valid]
            res["vdc"].append(vdc)
            res["avg_slope"].append(np.mean(slopes))
            res["std_slope"].append(np.std(slopes))
            res["avg_gain"].append(np.mean(gains))
            res["std_gain"].append(np.std(gains))
            res["per_seed"][vdc] = {"slopes": slopes, "gains": gains}
            print(f"    => Average Slope: {np.mean(slopes):.2e} +/- {np.std(slopes):.2e} (1/s) "
                  f"[{len(slopes)} valid]")
            print(f"    => Average Gain: {np.mean(gains):.2e} +/- {np.std(gains):.2e} electrons")

        results[label] = {k: (np.array(v) if k != "per_seed" else v) for k, v in res.items()}
    return results


def plot_debug(label, by_vdc, n_cols=3):
    """Current transients of every seed, with the fit points and lines, one panel per bias."""
    vdcs = sorted(by_vdc)
    n_rows = int(np.ceil(len(vdcs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle(label, fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for ax, vdc in zip(axes, vdcs):
        for seed in by_vdc[vdc]["skipped"]:
            ax.semilogy(seed["t"] * 1e12, seed["current"], color="grey", alpha=0.2, linewidth=1)
        for seed in by_vdc[vdc]["valid"]:
            t_fit = seed["t"][seed["mask_fit"]]
            ax.semilogy(seed["t"] * 1e12, seed["current"], alpha=0.3, linewidth=1)
            ax.semilogy(t_fit * 1e12, seed["current"][seed["mask_fit"]], "o", markersize=2)
            t_line = np.linspace(t_fit.min(), t_fit.max(), 100)
            ax.semilogy(t_line * 1e12, 10**(seed["slope"] * t_line + seed["intercept"]), "r--", linewidth=1)
        ax.axhline(y=I_MIN, color="g", linestyle=":", label=f"I_min={I_MIN:.0e}")
        ax.axhline(y=I_MAX, color="g", linestyle=":", label=f"I_max={I_MAX:.0e}")
        ax.axhline(y=I_AVALANCHE, color="r", linestyle="--", label=f"I_aval={I_AVALANCHE:.0e}")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Current (A)")
        ax.set_title(f"V = {vdc}V ({len(by_vdc[vdc]['skipped'])} skipped)")
        ax.grid(True)
        ax.legend(fontsize=8)
    for ax in axes[len(vdcs):]:
        ax.set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_vs_bias(results, key, ylabel, log=False):
    """Seed-averaged `key` ('slope' or 'gain') vs bias, one curve per prefix."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10.colors
    all_vdcs = set()
    for idx, (label, data) in enumerate(results.items()):
        all_vdcs.update(data["vdc"])
        ax.errorbar(data["vdc"], data[f"avg_{key}"], yerr=data[f"std_{key}"], fmt="o--",
                    markersize=10, capsize=5, capthick=2, elinewidth=2,
                    color=colors[idx % len(colors)], label=label)
    ax.set_xticks(sorted(all_vdcs))
    ax.set_xlabel("Bias Voltage (V)")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    plt.close("all")
    folders_by_prefix = find_seed_folders(BASE_PATH, SUBFOLDER_PREFIXES, VOLTAGES)
    seeds, t_max_min = load_seeds(BASE_PATH, folders_by_prefix)
    results = analyze(seeds, t_max_min)

    if DEBUG_PLOTS:
        for label, by_vdc in seeds.items():
            fig = plot_debug(label, by_vdc)
            save_figure(fig, os.path.join(SAVE_DIR, "debug_plots", f"debug_{label}"), dpi=200)

    if PLOT_GAIN:
        fig = plot_vs_bias(results, "gain", "Gain (electrons)", log=True)
        save_figure(fig, os.path.join(SAVE_DIR, "gain_vs_bias"), formats=("png", "pdf"), dpi=300)

    fig = plot_vs_bias(results, "slope", "Current growth rate (A/s)")
    save_figure(fig, os.path.join(SAVE_DIR, "slope_vs_bias"), formats=("png", "pdf"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
