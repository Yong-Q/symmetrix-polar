"""Shared matplotlib style for Symmetrix-Polar report figures.

Usage:
    from figstyle import apply, save, BLUE, GRAY, RED, INK, MUTED
    apply()
    ...
    save(fig, path)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One palette for the whole report.
BLUE = "#1f6feb"
GRAY = "#9aa4b2"
RED = "#d1495b"
GREEN = "#2da44e"
INK = "#24292f"
MUTED = "#57606a"
GRID = "#d8dee4"
LIGHT = "#f6f8fa"


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#8b949e",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
