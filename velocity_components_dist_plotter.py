import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool
import time
from scipy.constants import k, electron_mass, elementary_charge

class EnergyDistInput:
    def __init__(self, folder_path: str, file_name: str, save_prefix: str, save_folder_e: str, save_folder_h: str, id: int, dt: float, num_zoomed_bins: int = 100, min_zoomed_v: float = 6.0, max_zoomed_v: float = 6.0, num_count_bins: int = 100, min_count_v: float = 6.0,max_count_v: float = 6.0, max_zoomed: float = 2.0, max_count: float = 1e6):
        self.folder_path = folder_path
        self.file_name = file_name
        self.save_prefix = save_prefix
        self.save_folder_e = save_folder_e
        self.save_folder_h = save_folder_h
        self.id = id
        self.dt = dt
        self.num_zoomed_bins = num_zoomed_bins
        self.min_zoomed_v = min_zoomed_v
        self.max_zoomed_v = max_zoomed_v
        self.num_count_bins = num_count_bins
        self.min_count_v = min_count_v
        self.max_count_v = max_count_v
        self.max_zoomed = max_zoomed
        self.max_count = max_count
        



def create_energy_distribution_plots(input_data: EnergyDistInput):
    file = input_data.file_name     # Full path to the subhistory file
    save_folder_e = input_data.save_folder_e # Folder to save electron energy distribution plots
    save_folder_h = input_data.save_folder_h # Folder to save hole energy distribution plots
    save_prefix = input_data.save_prefix # Prefix for saved plot files
    id = input_data.id # Subhistory ID
    id_str = f"{id:04d}"  # Format ID as a zero-padded 4-digit string
    dt = input_data.dt # Time per subhistory. Will display time as dt*id

    max_zoomed_v = input_data.max_zoomed_v  # Maximum value for the zoomed plots
    max_count = input_data.max_count  # Maximum count for the count plots

    # Define bin edges for histograms
    edges = np.linspace(input_data.min_count_v, input_data.max_count_v, input_data.num_count_bins)  # 50 bins from 0 to 1e6

    # Create subplots
    fig_e, (ax1_e, ax2_e, ax3_e) = plt.subplots(3, 1, figsize=(6, 8))
    fig_h, (ax1_h, ax2_h, ax3_h) = plt.subplots(3, 1, figsize=(6, 8))

    # Read the first line to get number of electrons and holes
    with open(file, 'r') as f:
        first_line = f.readline().strip()
    
    sizes = first_line.split()

    num_eh = [int(sizes[1]), int(sizes[2])]  # Number of electrons and holes
    
    # Read the data matrix
    data = np.loadtxt(file, skiprows=1)
    
    # Extract electron and hole velocities
    if num_eh[0] > 0:

        # Gaussian (normal) distribution for a single velocity component at 240K
        T = 240  # Kelvin
        m = 0.36 * electron_mass # conduction mass
        v = np.linspace(input_data.min_count_v, input_data.max_count_v, 1000)
        sigma = np.sqrt(k * T / m)  # std dev for one component: <vx^2> = kT/m
        gauss = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (v ** 2) / (2 * sigma ** 2))
        # Normalize to peak = 1 to match histogram normalization used above
        # gauss = gauss / np.max(gauss)
        ax1_e.plot(v, gauss, 'r--', label=f'Gaussian {T}K ($v_x$)')

        vecs = data[:num_eh[0], 10:13]                       # shape (N,3)
        electron_velocity_x = vecs[:,0]  # vx
        electron_velocity_y = vecs[:,1]  # vy
        electron_velocity_z = vecs[:,2]  # vz

        # First subplot - Electron Velocity Distribution x
        counts, bins = np.histogram(electron_velocity_x, bins=edges, density=True)
        # weights = np.ones_like(electron_velocity_x)/(len(electron_velocity_x)*np.diff(edges)[0]*np.max(counts))
        ax1_e.hist(electron_velocity_x, bins=edges, color='b', edgecolor='k', alpha=0.7, density=True, label=f"$v_x$, |dE| = {abs((edges[1]-edges[0])):.2f} m/s")
        ax1_e.set_title(f'Normalized Electron Velocity Distribution, x component t = {dt*id:.2e}s')
        ax1_e.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        ax1_e.set_ylabel('Counts')
        ax1_e.set_xlabel('Velocity (m/s)')
        ax1_e.grid(True, alpha=0.3)
        ax1_e.legend()

        # Second subplot - Electron Velocity Distribution y
        counts, bins = np.histogram(electron_velocity_y, bins=edges, density=True)
        # weights = np.ones_like(electron_velocity_y)/(len(electron_velocity_y)*np.diff(edges)[0]*np.max(counts))
        ax2_e.hist(electron_velocity_y, bins=edges, color='b', edgecolor='k', alpha=0.7, density=True, label=f"$v_y$, |dE| = {abs((edges[1]-edges[0])):.2f} m/s")
        ax2_e.set_title(f'Normalized Electron Velocity Distribution, y component t = {dt*id:.2e}s')
        ax2_e.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        ax2_e.set_ylabel('Counts')
        ax2_e.set_xlabel('Velocity (m/s)')
        ax2_e.grid(True, alpha=0.3)
        ax2_e.plot(v, gauss, 'r--', label=f'Gaussian {T}K ($v_y$)')
        ax2_e.legend()

        # Third subplot - Electron Velocity Distribution z
        counts, bins = np.histogram(electron_velocity_z, bins=edges, density=True)
        # weights = np.ones_like(electron_velocity_z)/(len(electron_velocity_z)*np.diff(edges)[0]*np.max(counts))
        ax3_e.hist(electron_velocity_z, bins=edges, color='b', edgecolor='k', alpha=0.7, density=True, label=f"$v_z$, |dE| = {abs((edges[1]-edges[0])):.2f} m/s")
        ax3_e.set_title(f'Normalized Electron Velocity Distribution, z component t = {dt*id:.2e}s')
        ax3_e.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        ax3_e.set_ylabel('Counts')
        ax3_e.set_xlabel('Velocity (m/s)')
        ax3_e.grid(True, alpha=0.3)
        ax3_e.plot(v, gauss, 'r--', label=f'Gaussian {T}K ($v_z$)')
        ax3_e.legend()

        fig_e.tight_layout()
        fig_e.savefig(os.path.join(save_folder_e, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_e)
    
    if num_eh[1] > 0:
        
        # Gaussian (normal) distribution for a single velocity component at 240K (holes, effective mass)
        T = 240  # Kelvin
        v = np.linspace(input_data.min_count_v, input_data.max_count_v, 1000)
        # Heavy hole (hh) and light hole (lh) effective masses
        m_hh = 0.49 * electron_mass
        m_lh = 0.16 * electron_mass
        sigma_hh = np.sqrt(k * T / m_hh)
        sigma_lh = np.sqrt(k * T / m_lh)
        gauss_hh = (1.0 / (np.sqrt(2 * np.pi) * sigma_hh)) * np.exp(- (v ** 2) / (2 * sigma_hh ** 2))
        # gauss_lh = (1.0 / (np.sqrt(2 * np.pi) * sigma_lh)) * np.exp(- (v ** 2) / (2 * sigma_lh ** 2))
        # sum the two components and normalize peak to 1 for comparison with histogram
        gauss = gauss_hh 
        # gauss = gauss / np.max(gauss)

        vecs = data[num_eh[0]:num_eh[0]+num_eh[1], 10:13]   
        hole_velocity_x = vecs[:,0]  # vx
        hole_velocity_y = vecs[:,1]  # vy
        hole_velocity_z = vecs[:,2]  # vz

        # First subplot - Holes Velocity Distribution x
        counts, bins = np.histogram(hole_velocity_x, bins=edges, density=True)
        # weights = np.ones_like(hole_velocity_x)/(len(hole_velocity_x)*np.diff(edges)[0]*np.max(counts))
        ax1_h.hist(hole_velocity_x, bins=edges, color='r', edgecolor='k', alpha=0.7, density=True, label=f"$v_x$, |dE| = {abs((edges[1]-edges[0])):.2f} m/s")
        ax1_h.set_title(f'Normalized Hole Velocity Distribution, x component t = {dt*id:.2e}s')
        ax1_h.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        ax1_h.set_ylabel('Counts')
        ax1_h.set_xlabel('Velocity (m/s)')
        ax1_h.grid(True, alpha=0.3)
        ax1_h.plot(v, gauss, 'b--', label=f'Gaussian {T}K (hh+lh)')
        ax1_h.legend()

        # Second subplot - Holes Velocity Distribution y        
        counts, bins = np.histogram(hole_velocity_y, bins=edges, density=True)
        # weights = np.ones_like(hole_velocity_y)/(len(hole_velocity_y)*np.diff(edges)[0]*np.max(counts))
        ax2_h.hist(hole_velocity_y, bins=edges, color='r', edgecolor='k', alpha=0.7, density=True, label=f"$v_y$, |dE| = {abs((edges[1]-edges[0])):.2f} m/s")
        ax2_h.set_title(f'Normalized Hole Velocity Distribution, y component t = {dt*id:.2e}s')
        ax2_h.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        ax2_h.set_ylabel('Counts')
        ax2_h.set_xlabel('Velocity (m/s)')
        ax2_h.grid(True, alpha=0.3)
        ax2_h.plot(v, gauss, 'b--', label=f'Gaussian {T}K (hh+lh)')
        ax2_h.legend()

        # Third subplot - Holes Velocity Distribution z
        counts, bins = np.histogram(hole_velocity_z, bins=edges, density=True)
        # weights = np.ones_like(hole_velocity_z)/(len(hole_velocity_z)*np.diff(edges)[0]*np.max(counts))
        ax3_h.hist(hole_velocity_z, bins=edges, color='r', edgecolor='k', alpha=0.7, density=True, label=f"$v_z$, |dE| = {abs((edges[1]-edges[0])):.2f} m/s")
        ax3_h.set_title(f'Normalized Hole Velocity Distribution, z component t = {dt*id:.2e}s')
        ax3_h.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        ax3_h.set_ylabel('Counts')
        ax3_h.set_xlabel('Velocity (m/s)')
        ax3_h.grid(True, alpha=0.3)
        ax3_h.plot(v, gauss, 'b--', label=f'Gaussian {T}K (hh+lh)')
        ax3_h.legend()

        fig_h.tight_layout()
        fig_h.savefig(os.path.join(save_folder_h, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_e)




def main():
    # Parameters
    threads = 32 # Number of threads for parallel processing
    dt = 1e-12 # Time per subhistory in seconds

    base_dir = "/mnt/csmhome2/amudano/MonteCarlo/DEMC" # Folder containing device folders
    device_dir = "pn" # Device folder
    simulation_dir = "bias-0V-10fs-ab-HgCdTe" # Simulation folder

    folder = os.path.join(base_dir, device_dir, simulation_dir)
    
    save_folder_e = os.path.join(device_dir, simulation_dir, "velocity_frames_e") # Folder to save electron energy distribution plots
    save_folder_h = os.path.join(device_dir, simulation_dir, "velocity_frames_h") # Folder to save hole energy distribution plots
    prefix = "velocity_dist" # Prefix for saved plot files
    total_sh = 11 # Total number of subhistories

    num_zoomed_bins = 200
    min_zoomed_v = 5e5  # in m/s
    max_zoomed_v = 1e6  # in m/s
    num_count_bins = 200
    min_count_v = -1e6  # in m/s
    max_count_v = 1e6
    max_zoomed = 1e4
    max_count = 1e4

    # Create save directories if they don't exist
    os.makedirs(save_folder_e, exist_ok=True)
    os.makedirs(save_folder_h, exist_ok=True)

    time.sleep(1)


    input_arr = [EnergyDistInput(folder, os.path.join(folder, f"subhistory_{i}.txt"), prefix, save_folder_e, save_folder_h, i, dt, num_zoomed_bins, min_zoomed_v, max_zoomed_v, num_count_bins, min_count_v, max_count_v, max_zoomed, max_count) for i in range(total_sh)]


    print(f"Using {threads} threads for energy distribution plots.")
    print("This may take a while...")

    start = time.time()
    energy_pool = Pool(threads)
    energy_pool.map(create_energy_distribution_plots, input_arr)
    end = time.time()

    print(f"Velocity distribution plots created in {end - start:.2f} seconds.")

if __name__ == "__main__":
    main()

