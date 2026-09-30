"""Shared look for the literature-summary figures in docs/figures/.

Every figure script does `from _style import *` (run from this directory) and
draws with the named colours below, so all summaries share one visual language.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical (fixed order, never cycled): use in this order, at most three per figure.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
# Status: reserved for stable / unstable, never as an ordinary series colour.
GOOD = "#0ca30c"
BAD = "#d03b3b"
# Ink and chrome.
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
# Light fills for solid parts (wood, blocks): pair with an INK2 outline.
WOOD = "#e9dcc6"
WOOD2 = "#cdb691"
STONE = "#dedcd6"
# Sequential ramp (one hue, light -> dark) for magnitude.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def clean(ax, equal=False, hide_axes=False):
    """Recessive chrome: light grid, optional equal aspect, optional no axes."""
    if hide_axes:
        ax.set_axis_off()
    else:
        ax.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    return ax


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)
