"""Convert mp4 videos to colour-palette gifs, changing the playback speed.

Equivalent ffmpeg commands:
    ffmpeg -i foo.mp4 -filter:v "setpts=<1/multiplier>*PTS" foo_speed.mp4
    ffmpeg -i foo_speed.mp4 -vf palettegen palette.png
    ffmpeg -i foo_speed.mp4 -i palette.png -lavfi paletteuse foo.gif
"""
import os
import subprocess


def mp4_to_gif(video, multiplier):
    """Create <video>.gif from <video>.mp4; multiplier > 1 speeds up, < 1 slows down."""
    setpts = f"{1 / multiplier:.2f}"
    mp4_name = video + ".mp4"
    mp4_speed = video + "3.mp4"
    palette = video + "_palette.png"
    gif_name = video + ".gif"

    subprocess.run(["ffmpeg", "-i", mp4_name, "-filter:v", f"setpts={setpts}*PTS", mp4_speed], check=True)
    subprocess.run(["ffmpeg", "-i", mp4_speed, "-vf", "palettegen", palette], check=True)
    subprocess.run(["ffmpeg", "-i", mp4_speed, "-i", palette, "-lavfi", "paletteuse", gif_name], check=True)

    os.remove(mp4_speed)
    os.remove(palette)
    print("Finished creating gif:", gif_name)


def main():
    videos = [  # without the .mp4 extension
        "SPAD_FBK_v2/3D/4um/CONTINUE/m37V/-37V_Efield",
        "SPAD_FBK_v2/3D/4um/CONTINUE/m37V/-37V_Particles",
    ]
    multiplier = 0.5
    for video in videos:
        mp4_to_gif(video, multiplier)


if __name__ == "__main__":
    main()
