"""Matplotlib settings shared by every paper figure, matched to the MLSys template."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# The MLSys template sets 10 point Times in two 3.25 inch columns on a 6.75 inch text block.
COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.75
# The arms in the order of the chain in the paper, each with one Okabe-Ito colour, which stays
# distinct under colour blindness, and one marker in every figure.
ARMS = (
    ("per_layer", "Decoder-only", "#0072B2"),
    ("ced_full", "CED", "#CC79A7"),
    ("ced", "CED-SWA", "#009E73"),
    ("shared_cache", "DD", "#D55E00"),
)
# Bar fills and hatches for the four designs in arm order, shared by every bar figure. Every fill
# stays light enough that black error bars show over it.
ARM_FILLS = ("#8c8c8c", "white", "#d9d9d9", "#d9d9d9")
ARM_HATCHES = ("", "////", "", "\\\\\\\\")
DESIGN_BAR_WIDTH = 0.18
MARKERS = {"per_layer": "o", "ced_full": "D", "ced": "^", "shared_cache": "s"}
# Every figure prints in grayscale, like the first figure. Decoder-only and DD, the two ends
# of the comparison, are black. CED and CED-SWA are gray. Line style separates each pair.
LINE_STYLES = {
    "per_layer": {"color": "black", "linestyle": "-", "marker": "o"},
    "ced_full": {"color": "#8c8c8c", "linestyle": "-", "marker": "D"},
    "ced": {"color": "#8c8c8c", "linestyle": "--", "marker": "^"},
    "shared_cache": {"color": "black", "linestyle": "--", "marker": "s"},
}
GRID_COLOR = "#dddddd"
BAND_COLOR = "#eeeeee"
LEGEND_COLOR = "#444444"


def apply_style() -> None:
    """Set fonts, sizes, and line weights for a figure printed at its final size."""
    mpl.rcParams.update(
        {
            # Submission checks require every font to be embedded. Times New Roman is a TrueType
            # font that matplotlib embeds as a subset, and it matches the paper's Times body text.
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            # The core fonts carry no Unicode minus sign, so negative ticks use a hyphen.
            "axes.unicode_minus": False,
            # 8 points is the smallest size that stays legible in print, so every label uses it.
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "errorbar.capsize": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def draw_design_bars(
    axis: Axes,
    group_labels: tuple[str, ...],
    means: np.ndarray,
    spreads: np.ndarray | None = None,
) -> None:
    """One group of bars per label, with one bar per design in arm order.

    means and spreads: (groups, designs). Spreads, when given, draw as error bars.
    """
    for group_index in range(len(group_labels)):
        for arm_index in range(len(ARMS)):
            error = None if spreads is None else spreads[group_index, arm_index]
            axis.bar(
                group_index + (arm_index - 1.5) * DESIGN_BAR_WIDTH,
                means[group_index, arm_index],
                width=DESIGN_BAR_WIDTH * 0.9,
                color=ARM_FILLS[arm_index],
                hatch=ARM_HATCHES[arm_index],
                edgecolor="black",
                linewidth=0.6,
                yerr=error,
                error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "black"},
                label=ARMS[arm_index][1] if group_index == 0 else None,
            )
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_xticks(range(len(group_labels)), group_labels)
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.5, len(group_labels) - 0.5)


def design_legend(figure: Figure, axis: Axes) -> None:
    """One legend row above the figure naming the four designs, as in the line figures."""
    handles = axis.get_legend_handles_labels()
    figure.legend(
        *handles,
        loc="upper center",
        ncol=len(ARMS),
        frameon=False,
        columnspacing=0.6,
        handlelength=0.9,
        handletextpad=0.3,
    )
