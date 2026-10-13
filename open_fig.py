"""Open editable figures (.mplfig) saved by mcplot.save_figure.

Usage:
    python open_fig.py figure.mplfig [other.mplfig ...]

In the window (Qt backend, installed with the MCPlot environment):
  - toolbar button "Edit axis, curve and image parameters": titles, axis
    labels, limits, scales, line labels/style/width/colour, markers;
  - "s" / Ctrl+S or the save button: export to png, pdf, svg, ...;
  - Shift+S: overwrite the .mplfig with the edited figure.

For edits not covered by the dialog (fonts, histogram bars, legends, ...)
load the figure in Python with `mcplot.load_figure` and modify it there.
"""
import argparse
import pickle

import matplotlib.pyplot as plt

from mcplot import load_figure


def _add_resave_key(fig, path):
    def on_key(event):
        if event.key == "S":
            with open(path, "wb") as f:
                pickle.dump(fig, f)
            print(f"Saved editable figure: {path}")

    fig.canvas.mpl_connect("key_press_event", on_key)


def main():
    parser = argparse.ArgumentParser(description="Open editable .mplfig figures.")
    parser.add_argument("files", nargs="+", help=".mplfig files to open")
    args = parser.parse_args()

    for path in args.files:
        fig = load_figure(path)
        _add_resave_key(fig, path if path.endswith(".mplfig") else path + ".mplfig")
    plt.show()


if __name__ == "__main__":
    main()
