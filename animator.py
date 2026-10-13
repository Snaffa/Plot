"""Build an mp4 movie from the .png frames of a folder (sorted by name)."""
import os

import cv2


def create_movie_from_images(image_folder, output_file, fps, frame_size=None):
    images = sorted(img for img in os.listdir(image_folder) if img.endswith(".png"))
    if not images:
        raise ValueError(f"No PNG images found in {image_folder}.")

    # Use the size of the first image if frame_size is not provided
    if frame_size is None:
        height, width = cv2.imread(os.path.join(image_folder, images[0])).shape[:2]
        frame_size = (width, height)

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        if frame is None:
            print(f"Warning: Could not read image {image}. Skipping.")
            continue
        if frame.shape[1::-1] != frame_size:
            frame = cv2.resize(frame, frame_size)
        video.write(frame)
    video.release()
    print(f"Movie created successfully: {output_file}")


def main():
    image_folder = "SPAD_FBK_v2/quasi_1D/CONTINUE/quenching--37.00V/velocity_frames_e"
    output_file = "SPAD_FBK_v2/quasi_1D/CONTINUE/quenching--37.00V/velocity_frames_e.mp4"
    fps = 10
    create_movie_from_images(image_folder, output_file, fps)


if __name__ == "__main__":
    main()
