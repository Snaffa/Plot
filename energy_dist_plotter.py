import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool
import time
import cv2
import subprocess

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

    # Define bin edges for histograms
    edges = np.linspace(0, input_data.max_count_ev, input_data.num_count_bins)  # 50 bins from 0 to 6
    prob_edges = np.linspace(0, input_data.max_pdf_ev, input_data.num_pdf_bins)  # 50 bins from 0 to 6

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
    
    # Extract electron and hole energies
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

        # Second subplot - Normalized Electron Energy Distribution
        counts, bins = np.histogram(electron_energy, bins=prob_edges, density=True)
        weights = np.ones_like(electron_energy)/(len(electron_energy)*np.diff(prob_edges)[0]*np.max(counts))
        ax2_e.hist(electron_energy, bins=prob_edges, color='b', edgecolor='k', alpha=0.7, weights=weights,label=f'dE = {(edges[1]-edges[0])/1e-3:.2f} meV')
        ax2_e.set_title(f'Normalized Electron Energy Distribution t = {dt*id:.2e}s')
        # ax2_e.set_ylim([1e-4, max_pdf])  # Start from 1e-4 for log scale
        ax2_e.set_ylabel('Probability Density')
        ax2_e.set_xlabel('Energy (eV)')
        ax2_e.grid(True, alpha=0.3)

        # Maxwell energy distribution (3D) at 300 K, in eV:
        kB_eV = 8.617333262e-5  # eV / K
        T = 300  # Kelvin
        kT = kB_eV * T
        centers = 0.5 * (prob_edges[:-1] + prob_edges[1:])
        # f_E(E) = 2/sqrt(pi) * E^{1/2} / (kT)^{3/2} * exp(-E/kT)
        maxwell = (2.0 / np.sqrt(np.pi)) * (centers ** 0.5) / (kT ** 1.5) * np.exp(-centers / kT)
        # scale to peak = 1 to match the histogram normalization used above
        if np.max(maxwell) > 0:
            maxwell /= np.max(maxwell)
        ax2_e.plot(centers, maxwell, 'r--', label=f'Maxwell {T}K')
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

        # Second subplot - Normalized Hole Energy Distribution
        counts, bins = np.histogram(hole_energy, bins=prob_edges, density=True)
        weights = np.ones_like(hole_energy)/(len(hole_energy)*np.diff(prob_edges)[0]*np.max(counts))
        ax2_h.hist(hole_energy, bins=prob_edges, color='r', edgecolor='k', alpha=0.7, weights=weights,label=f'dE = {(prob_edges[1]-prob_edges[0])/1e-3:.2f} meV')
        ax2_h.set_title(f'Normalized Hole Energy Distribution t = {dt*id:.2e}s')
        ax2_h.set_ylabel('Probability Density')
        ax2_h.set_xlabel('Energy (eV)')
        ax2_h.grid(True, alpha=0.3)
        # Maxwell energy distribution (3D) at 240 K, in eV:
        centers_h = 0.5 * (prob_edges[:-1] + prob_edges[1:])
        maxwell_h = (2.0 / np.sqrt(np.pi)) * (centers_h ** 0.5) / (kT ** 1.5) * np.exp(-centers_h / kT)
        if np.max(maxwell_h) > 0:
            maxwell_h /= np.max(maxwell_h)
        ax2_h.plot(centers_h, maxwell_h, 'b--', label=f'Maxwell {T}K')
        ax2_h.legend(loc='upper right')


        fig_h.tight_layout()
        fig_h.savefig(os.path.join(save_folder_h, f'{save_prefix}_{id_str}'), dpi=200)
        # plt.close(fig_h)


def create_movie_from_images(image_folder, output_file, fps, frame_size=None):
    # Get all image file names in the folder, sorted
    images = [img for img in sorted(os.listdir(image_folder)) if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg"))]
    
    if not images:
        raise ValueError("No JPEG images found in the specified folder.")
    
    # Get the size of the first image if frame_size is not provided
    if frame_size is None:
        first_image_path = os.path.join(image_folder, images[0])
        first_frame = cv2.imread(first_image_path)
        height, width, layers = first_frame.shape
        frame_size = (width, height)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'MJPG' for other formats
    video = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # Write each image as a frame into the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {image}. Skipping.")
            continue
        
        # Ensure all frames are of the same size
        if frame.shape[1::-1] != frame_size:
            frame = cv2.resize(frame, frame_size)

        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Movie created successfully: {output_file}")


def create_gif_from_mp4(video, multiplier):
    #this script takes the .mp4 file name and speeds it up by 3x then converts it to a gif with colors

    '''
        The following ffmpeg commands are used to create the gif:
        ffmpeg -i foo.mp4 -filter:v "setpts=0.33*PTS" foo3.mp4
        ffmpeg -i ef2.mp4 -vf palettegen palette.png
        ffmpeg -i ef2.mp4 -i palette.png -lavfi "paletteuse"  ef.gif
    '''


    speed = str(round(1/multiplier, 2))
    
    mp4_name = video + ".mp4"
    mp4_name_speed = video + "3.mp4"
    gif_name = video + ".gif"
    palette_name = video + "_palette.png"

    #speed up the video
    command = "ffmpeg -i " + video + " -filter:v 'setpts=" + speed + "*PTS' " + mp4_name_speed
    p = subprocess.Popen([command], shell=True)
    p.wait()

    #create the palette
    command = "ffmpeg -i " + mp4_name_speed + " -vf palettegen " + palette_name
    p = subprocess.Popen([command], shell=True)
    p.wait()

    #create the gif
    command = "ffmpeg -i " + mp4_name_speed + " -i " + palette_name + " -lavfi 'paletteuse' " + gif_name
    p = subprocess.Popen([command], shell=True)
    p.wait()

    
    
    #remove the speed up video
    p = subprocess.Popen(["rm " + mp4_name_speed], shell=True)
    p.wait()

    p = subprocess.Popen(["rm " + palette_name], shell=True)
    p.wait()

    print("Finished creating gif: ", gif_name)


def main():
    # Parameters
    threads = 15 # Number of threads for parallel processing
    dt = 10e-12 # Time per subhistory in seconds

    base_dir = "/mnt/polmcad/MonteCarlo/DEMC" # Folder containing device folders
    device_dir = "prove" # Device folder
    simulation_dir = "bias/eq" # Simulation folder

    folder = os.path.join(base_dir, device_dir, simulation_dir)
    
    save_folder_e = os.path.join(device_dir, simulation_dir, "energy_frames_e") # Folder to save electron energy distribution plots
    save_folder_h = os.path.join(device_dir, simulation_dir, "energy_frames_h") # Folder to save hole energy distribution plots
   
    prefix = "energy_dist" # Prefix for saved plot files
    total_sh = 11 # Total number of subhistories

    num_pdf_bins = 200 
    max_pdf_ev = 0.5
    num_count_bins = 200
    max_count_ev = 0.5
    max_pdf = 1
    max_count = 1e6

    # Create save directories if they don't exist
    os.makedirs("/home/angelo", exist_ok=True)
    os.makedirs("/home/angelo", exist_ok=True)

    time.sleep(1)


    input_arr = [EnergyDistInput(folder, os.path.join(folder, f"subhistory_{i}.txt"), prefix, save_folder_e, save_folder_h, i, dt, num_pdf_bins, max_pdf_ev, num_count_bins, max_count_ev, max_pdf, max_count) for i in range(total_sh)]


    print(f"Using {threads} threads for energy distribution plots.")
    print("This may take a while...")

    start = time.time()
    energy_pool = Pool(threads)
    energy_pool.map(create_energy_distribution_plots, input_arr)
    end = time.time()

    print(f"Energy distribution plots created in {end - start:.2f} seconds.")
    # Create movie from saved images
    for folder in [save_folder_e, save_folder_h]:
        image_folder = folder  # Folder containing images
        print("Creating movie from images in folder:", image_folder)
        output_name = "energy_distribution_movie"
        print("Output movie name:", output_name)
        output_file = folder + "/" + output_name + ".mp4"  # Output movie file path
        fps = 30  # Frames per second
        create_movie_from_images(image_folder, output_file, fps)
        # create_gif_from_mp4(output_name,multiplier=3)

if __name__ == "__main__":
    main()



