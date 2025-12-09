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
    zoomed_edges = np.linspace(input_data.min_zoomed_v, input_data.max_zoomed_v, input_data.num_zoomed_bins)  # 50 bins from 0 to 6

    # Create subplots
    fig_e, (ax1_e, ax2_e) = plt.subplots(2, 1, figsize=(6, 8))
    fig_h, (ax1_h, ax2_h) = plt.subplots(2, 1, figsize=(6, 8))

    # Read the first line to get number of electrons and holes
    with open(file, 'r') as f:
        first_line = f.readline().strip()
    
    sizes = first_line.split()

    num_eh = [int(sizes[1]), int(sizes[2])]  # Number of electrons and holes
    
    # Read the data matrix
    data = np.loadtxt(file, skiprows=1)
    
    # Extract electron and hole velocities
    if num_eh[0] > 0:
        vecs = data[:num_eh[0], 10:13]                       # shape (N,3)
        electron_velocity = np.linalg.norm(vecs, axis=1)    # moduli, shape (N,)
        # electron_velocity = np.sqrt(data[:num_eh[0], 10]**2 + data[:num_eh[0], 11]**2 + data[:num_eh[0], 12]**2)  # 4th column (0-indexed as 3)
        
        # First subplot - Electron Velocity Distribution
        counts, bins = np.histogram(electron_velocity, bins=edges, density=True)
        weights = np.ones_like(electron_velocity)/(len(electron_velocity)*np.diff(edges)[0]*np.max(counts))
        ax1_e.hist(electron_velocity, bins=edges, color='b', edgecolor='k', alpha=0.7, weights=weights, label=f"dE = {(edges[1]-edges[0]):.2f} m/s")

        # ax1_e.hist(electron_velocity, bins=edges, edgecolor='k', color='b', alpha=0.7,density=True)
        ax1_e.set_title(f'Normalized Electron Velocity Distribution t = {dt*id:.2e}s')
        ax1_e.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        # ax1_e.set_yscale('log')
        ax1_e.set_ylabel('Counts')
        ax1_e.set_xlabel('Velocity (m/s)')
        # ax1_e.set_xscale('log')
        ax1_e.grid(True, alpha=0.3)
        
        # Maxwell-Boltzmann velocity distribution at 300K
        T = 300  # Kelvin
        m = electron_mass
        v = np.linspace(input_data.min_count_v, input_data.max_count_v, 1000)
        maxwell = 4*np.pi * (m/(2*np.pi*k*T))**(3/2) * v**2 * np.exp(-m*v**2/(2*k*T))
        # Normalize to peak = 1
        maxwell = maxwell/np.max(maxwell)
        ax1_e.plot(v, maxwell, 'r--', label='Maxwell 300K')
        ax1_e.legend()

        # Second subplot - Electron Velocity Distribution zoomed
        counts, bins = np.histogram(electron_velocity, bins=zoomed_edges, density=True)
        weights = np.ones_like(electron_velocity)/(len(electron_velocity)*np.diff(zoomed_edges)[0]*np.max(counts))
        ax2_e.hist(electron_velocity, bins=zoomed_edges, color='b', edgecolor='k', alpha=0.7, weights=weights,label=f'dE = {(zoomed_edges[1]-zoomed_edges[0]):.2f} m/s')
        ax2_e.set_title(f'Normalized Electron Velocity Distribution zoom t = {dt*id:.2e}s')
        ax2_e.set_xlim([input_data.min_zoomed_v, input_data.max_zoomed_v])  # Start from 1 for log scale
        ax2_e.set_ylim([0, 0.1])  # Start from 1 for log scale
        # ax2_e.set_yscale('log')
        ax2_e.set_ylabel('Counts')
        ax2_e.set_xlabel('Velocity (m/s)')
        ax2_e.grid(True, alpha=0.3)
        ax2_e.legend()

        fig_e.tight_layout()
        fig_e.savefig(os.path.join(save_folder_e, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_e)
    
    if num_eh[1] > 0:
        vecs = data[num_eh[0]:num_eh[0]+num_eh[1], 10:13]                       # shape (N,3)
        hole_velocity = np.linalg.norm(vecs, axis=1)    # moduli, shape (N,)
        # hole_energy = data[num_eh[0]:num_eh[0]+num_eh[1], 3]  # 4th column (0-indexed as 3)

        # First subplot - Hole Velocity Distribution
        counts, bins = np.histogram(hole_velocity, bins=edges, density=True)
        weights = np.ones_like(hole_velocity)/(len(hole_velocity)*np.diff(edges)[0]*np.max(counts))
        ax1_h.hist(hole_velocity, bins=edges, color='r', edgecolor='k', alpha=0.7, weights=weights,label=f'dE = {(edges[1]-edges[0]):.2f} m/s')
        ax1_h.set_title(f'Normalized Hole Velocity Distribution t = {dt*id:.2e}s')
        ax1_h.set_xlim([input_data.min_count_v, input_data.max_count_v])  # Start from 1 for log scale
        # ax1_h.set_ylim([1, max_count])  # Start from 1 for log scale
        # ax1_h.set_yscale('log')
        ax1_h.set_ylabel('Counts')
        ax1_h.set_xlabel('Velocity (m/s)')
        # ax1_h.set_xscale('log')
        ax1_h.grid(True, alpha=0.3)
        
        # Maxwell-Boltzmann velocity distribution at 300K
        T = 300  # Kelvin
        m = 0.36 * electron_mass  # hole effective mass
        v = np.linspace(input_data.min_count_v, input_data.max_count_v, 1000)
        maxwell = 4*np.pi * (m/(2*np.pi*k*T))**(3/2) * v**2 * np.exp(-m*v**2/(2*k*T))
        # Normalize to peak = 1
        maxwell = maxwell/np.max(maxwell)
        ax1_h.plot(v, maxwell, 'b--', label='Maxwell 300K')
        ax1_h.legend()

        # Second subplot - Hole Velocity Distribution zoomed
        counts, bins = np.histogram(hole_velocity, bins=zoomed_edges, density=True)
        weights = np.ones_like(hole_velocity)/(len(hole_velocity)*np.diff(zoomed_edges)[0]*np.max(counts))
        ax2_h.hist(hole_velocity, bins=zoomed_edges, color='r', edgecolor='k', alpha=0.7, weights=weights,label=f'dE = {(zoomed_edges[1]-zoomed_edges[0]):.2f} m/s')
        ax2_h.set_title(f'Normalized Hole Velocity Distribution zoom t = {dt*id:.2e}s')
        ax2_h.set_xlim([input_data.min_zoomed_v, input_data.max_zoomed_v])  # Start from 1 for log scale
        # ax2_h.set_ylim([1, max_count])  # Start from 1 for log scale
        # ax2_h.set_yscale('log')
        ax2_h.set_ylabel('Counts')
        ax2_h.set_xlabel('Velocity (m/s)')
        ax2_h.grid(True, alpha=0.3)
        ax2_h.set_ylim([0, 0.5])  # Start from 1 for log scale
        ax2_h.legend()

        fig_h.tight_layout()
        fig_h.savefig(os.path.join(save_folder_h, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_h)




def main():
    # Parameters
    threads = 32 # Number of threads for parallel processing
    dt = 1e-12 # Time per subhistory in seconds

    base_dir = "P:/mnt/polito/MonteCarlo/DEMC" # Folder containing device folders
    device_dir = "pn" # Device folder
    simulation_dir = "bias-0V-10fs-fb-mio" # Simulation folder

    folder = os.path.join(base_dir, device_dir, simulation_dir)

    save_folder_e = os.path.join(device_dir, simulation_dir, "velocity_modulus_frames_e") # Folder to save electron energy distribution plots
    save_folder_h = os.path.join(device_dir, simulation_dir, "velocity_modulus_frames_h") # Folder to save hole energy distribution plots
    prefix = "velocity_modulus_dist" # Prefix for saved plot files
    total_sh = 11

    num_zoomed_bins = 100
    min_zoomed_v = 5e5  # in m/s
    max_zoomed_v = 1e6  # in m/s
    num_count_bins = 100
    min_count_v = 0  # in m/s
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

