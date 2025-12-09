import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool
import time
from scipy.optimize import curve_fit

# Physical constant: Boltzmann in eV/K
kB_eV = 8.617333262e-5  # eV / K

class EnergyDistInput:
    def __init__(self, folder_path: str, file_name: str, save_prefix: str, save_folder_e: str, save_folder_h: str, id: int, dt: float, num_pdf_bins: int = 100, max_pdf_ev: float = 6.0, num_count_bins: int = 100, max_count_ev: float = 6.0, max_pdf: float = 2.0, max_count: float = 1e6):
        self.folder_path = folder_path
        self.file_name = file_name
        self.save_prefix = save_prefix
        self.save_folder_e = save_folder_e
        self.save_folder_h = save_folder_h
        self.id = id
        self.dt = dt
        self.num_pdf_bins = num_pdf_bins 
        self.max_pdf_ev = max_pdf_ev
        self.num_count_bins = num_count_bins
        self.max_count_ev = max_count_ev
        self.max_pdf = max_pdf
        self.max_count = max_count


def _normalized_maxwell_energy(E, A, kT):
    # model(E) = A * sqrt(E) * exp(-E/kT); returned normalized to peak = 1
    Epos = np.clip(E, 1e-12, None)
    m = A * np.sqrt(Epos) * np.exp(-Epos / kT)
    maxm = np.max(m)
    if maxm <= 0:
        return m
    return m / maxm


def create_energy_distribution_plots(input_data: EnergyDistInput):
    file = input_data.file_name     # Full path to the subhistory file
    save_folder_e = input_data.save_folder_e # Folder to save electron energy distribution plots
    save_folder_h = input_data.save_folder_h # Folder to save hole energy distribution plots
    save_prefix = input_data.save_prefix # Prefix for saved plot files
    id = input_data.id # Subhistory ID
    id_str = f"{id:04d}"  # Format ID as a zero-padded 4-digit string
    dt = input_data.dt # Time per subhistory. Will display time as dt*id

    max_pdf = input_data.max_pdf  # Maximum value for the probability density function plots
    max_count = input_data.max_count  # Maximum count for the count plots

    # Define bin edges for histograms (in eV)
    edges = np.linspace(0, input_data.max_count_ev, input_data.num_count_bins)
    prob_edges = np.linspace(0, input_data.max_pdf_ev, input_data.num_pdf_bins)

    # Prepare figures
    fig_e, (ax1_e, ax2_e) = plt.subplots(2, 1, figsize=(6, 8))
    fig_h, (ax1_h, ax2_h) = plt.subplots(2, 1, figsize=(6, 8))

    # Read the first line to get number of electrons and holes
    with open(file, 'r') as f:
        first_line = f.readline().strip()
    sizes = first_line.split()

    num_eh = [int(sizes[1]), int(sizes[2])]  # Number of electrons and holes
    
    # Read the data matrix
    data = np.loadtxt(file, skiprows=1)

    Te_e = None
    Te_h = None

    # ELECTRONS
    if num_eh[0] > 0:
        electron_energy = data[:num_eh[0], 3]  # 4th column (0-indexed as 3)
        
        
        # First subplot - Electron Energy Distribution
        ax1_e.hist(electron_energy, bins=edges, edgecolor='k', color='b', alpha=0.7,label=f'dE = {(edges[1]-edges[0])/1e-3:.2f} meV')
        ax1_e.set_title(f'Electron Energy Distribution t = {dt*id:.2e}s')
        ax1_e.set_ylim([0.8, max_count])  # Start from 1 for log scale
        ax1_e.set_yscale('log')
        ax1_e.set_ylabel('Counts')
        ax1_e.set_xlabel('Energy (eV)')
        ax1_e.grid(True, alpha=0.3)
        ax1_e.legend(loc='upper right')

        # Normalized PDF (peak = 1)
        counts, bins = np.histogram(electron_energy, bins=prob_edges, density=True)
        if np.max(counts) > 0:
            norm_counts = counts / np.max(counts)
        else:
            norm_counts = counts

        centers = 0.5 * (bins[:-1] + bins[1:])

        # Fit normalized Maxwellian: fit A and kT (kT in eV)
        try:
            p0 = [1.0, kB_eV * 300.0]
            bounds = ([0.0, 1e-8], [np.inf, np.inf])
            popt, _ = curve_fit(_normalized_maxwell_energy, centers, norm_counts, p0=p0, bounds=bounds, maxfev=5000)
            A_fit, kT_fit = popt
            Te_e = float(kT_fit / kB_eV)
            fitted = _normalized_maxwell_energy(centers, A_fit, kT_fit)
            ax2_e.hist(electron_energy, bins=prob_edges, color='b', edgecolor='k', alpha=0.7, weights=np.ones_like(electron_energy)/(len(electron_energy)*np.diff(prob_edges)[0]*np.max(counts)),label=f'dE = {(prob_edges[1]-prob_edges[0])/1e-3:.2f} meV')
            ax2_e.plot(centers, fitted, 'r--', label=f'Maxwell fit: T_eq={Te_e:.1f} K')
            ax2_e.set_title(f'Normalized Electron Energy Distribution t = {dt*id:.2e}s')
            ax2_e.set_ylabel('Probability Density (peak=1)')
            ax2_e.set_xlabel('Energy (eV)')
            ax2_e.grid(True, alpha=0.3)
            ax2_e.legend(loc='upper right')
        except Exception:
            # If fit fails, plot 300K reference normalized
            kT_ref = kB_eV * 300.0
            maxwell_ref = (2.0 / np.sqrt(np.pi)) * (centers ** 0.5) / (kT_ref ** 1.5) * np.exp(-centers / kT_ref)
            if np.max(maxwell_ref) > 0:
                maxwell_ref /= np.max(maxwell_ref)
            ax2_e.hist(electron_energy, bins=prob_edges, color='b', edgecolor='k', alpha=0.7, weights=np.ones_like(electron_energy)/(len(electron_energy)*np.diff(prob_edges)[0]*np.max(counts)))
            ax2_e.plot(centers, maxwell_ref, 'r--', label='Maxwell 300K (ref)')
            ax2_e.legend(loc='upper right')

        fig_e.tight_layout()
        fig_e.savefig(os.path.join(save_folder_e, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_e)
    
    if num_eh[1] > 0:
        hole_energy = data[num_eh[0]:num_eh[0]+num_eh[1], 3]  # 4th column (0-indexed as 3)

        # First subplot - Hole Energy Distribution
        ax1_h.hist(hole_energy, bins=edges, edgecolor='k', color='r', alpha=0.7,label=f'dE = {(edges[1]-edges[0])/1e-3:.2f} meV')
        ax1_h.set_title(f'Hole Energy Distribution t = {dt*id:.2e}s')
        ax1_h.set_ylim([0.8, max_count])  # Start from 1 for log scale
        ax1_h.set_yscale('log')
        ax1_h.set_ylabel('Counts')
        ax1_h.set_xlabel('Energy (eV)')
        ax1_h.grid(True, alpha=0.3)
        ax1_h.legend(loc='upper right')

        counts_h, bins_h = np.histogram(hole_energy, bins=prob_edges, density=True)
        if np.max(counts_h) > 0:
            norm_counts_h = counts_h / np.max(counts_h)
        else:
            norm_counts_h = counts_h

        centers_h = 0.5 * (bins_h[:-1] + bins_h[1:])

        # Fit normalized Maxwellian for holes
        try:
            p0 = [1.0, kB_eV * 300.0]
            bounds = ([0.0, 1e-8], [np.inf, np.inf])
            popt_h, _ = curve_fit(_normalized_maxwell_energy, centers_h, norm_counts_h, p0=p0, bounds=bounds, maxfev=5000)
            A_fit_h, kT_fit_h = popt_h
            Te_h = float(kT_fit_h / kB_eV)
            fitted_h = _normalized_maxwell_energy(centers_h, A_fit_h, kT_fit_h)
            ax2_h.hist(hole_energy, bins=prob_edges, color='r', edgecolor='k', alpha=0.7, weights=np.ones_like(hole_energy)/(len(hole_energy)*np.diff(prob_edges)[0]*np.max(counts_h)),label=f'dE = {(prob_edges[1]-prob_edges[0])/1e-3:.2f} meV')
            ax2_h.plot(centers_h, fitted_h, 'b--', label=f'Maxwell fit: T_eq={Te_h:.1f} K')
            ax2_h.set_title(f'Normalized Hole Energy Distribution t = {dt*id:.2e}s')
            ax2_h.set_ylabel('Probability Density (peak=1)')
            ax2_h.set_xlabel('Energy (eV)')
            ax2_h.grid(True, alpha=0.3)
            ax2_h.legend(loc='upper right')
        except Exception:
            kT_ref = kB_eV * 300.0
            maxwell_ref_h = (2.0 / np.sqrt(np.pi)) * (centers_h ** 0.5) / (kT_ref ** 1.5) * np.exp(-centers_h / kT_ref)
            if np.max(maxwell_ref_h) > 0:
                maxwell_ref_h /= np.max(maxwell_ref_h)
            ax2_h.hist(hole_energy, bins=prob_edges, color='r', edgecolor='k', alpha=0.7, weights=np.ones_like(hole_energy)/(len(hole_energy)*np.diff(prob_edges)[0]*np.max(counts_h)))
            ax2_h.plot(centers_h, maxwell_ref_h, 'b--', label='Maxwell 300K (ref)')
            ax2_h.legend(loc='upper right')


        fig_h.tight_layout()
        fig_h.savefig(os.path.join(save_folder_h, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_h)

    # return tuple (id, Te_e, Te_h) for aggregation in main
    return (id, Te_e, Te_h)


def main():
    # Parameters
    threads = 32 # Number of threads for parallel processing
    dt = 1e-12 # Time per subhistory in seconds

    folder = "P:/mnt/polito/MonteCarlo/DEMC/SPAD_FBK_v2/quasi_2D/half/bulk/CONTINUE-m18-Vov-Vq_analytic-variable_dt-150ps" # Folder containing subhistory files
    save_folder_e = "CONTINUE-m18-Vov-Vq_analytic-variable_dt-150ps/energy_frames_e" # Folder to save electron energy distribution plots
    save_folder_h = "CONTINUE-m18-Vov-Vq_analytic-variable_dt-150ps/energy_frames_h" # Folder to save hole energy distribution plots
    save_folder_T = "CONTINUE-m18-Vov-Vq_analytic-variable_dt-150ps/equivalent_temperature" # Folder to save equivalent temperature plots
    prefix = "energy_dist" # Prefix for saved plot files
    total_sh = 151

    num_pdf_bins = 200 
    max_pdf_ev = 2
    num_count_bins = 200
    max_count_ev = 2
    max_pdf = 1
    max_count = 1e4

    # Create save directories if they don't exist
    os.makedirs(save_folder_e, exist_ok=True)
    os.makedirs(save_folder_h, exist_ok=True)

    time.sleep(1)


    input_arr = [EnergyDistInput(folder, os.path.join(folder, f"subhistory_{i}.txt"), prefix, save_folder_e, save_folder_h, i, dt, num_pdf_bins, max_pdf_ev, num_count_bins, max_count_ev, max_pdf, max_count) for i in range(total_sh)]

    print(f"Using {threads} threads for energy distribution plots.")
    print("This may take a while...")

    start = time.time()
    with Pool(threads) as energy_pool:
        results = energy_pool.map(create_energy_distribution_plots, input_arr)
    end = time.time()

    print(f"Energy distribution plots created in {end - start:.2f} seconds.")

    # Aggregate temperatures
    Te_e_arr = np.full(total_sh, np.nan)
    Te_h_arr = np.full(total_sh, np.nan)
    for r in results:
        if r is None:
            continue
        idx, Te_e, Te_h = r
        Te_e_arr[idx] = Te_e if Te_e is not None else np.nan
        Te_h_arr[idx] = Te_h if Te_h is not None else np.nan

    # Time axis: linspace from 0 to (N-1)*dt
    t = np.linspace(0.0, (total_sh-1)*dt, total_sh)

    # Plot temperatures vs time
    plt.figure(figsize=(8,4))
    plt.plot(t, Te_e_arr, 'o-', label='Electrons')
    plt.plot(t, Te_h_arr, 's-', label='Holes')
    plt.xlabel('Time (s)')
    plt.ylabel('Equivalent temperature (K)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_timeplot = os.path.join(save_folder_T, f'{prefix}_Te_vs_time.png')
    plt.savefig(out_timeplot, dpi=200)
    plt.close()

    print(f"Saved T_eq vs time to {out_timeplot}")


if __name__ == "__main__":
    main()

