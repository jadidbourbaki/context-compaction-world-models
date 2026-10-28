"""Build the compaction figure for the paper: raw window against compacted context, per arm."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
    ARMS,
    BAND_COLOR,
    GRID_COLOR,
    TEXT_WIDTH,
    apply_style,
)
from matplotlib.lines import Line2D
from pydantic import BaseModel, DirectoryPath, Field

SEEDS = (42, 43, 44)
GRAY = "#8c8c8c"
GAP_LABELS = ("1-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511")


class Settings(BaseModel):
    """Paths used to construct the compaction figure."""

    results: DirectoryPath = Field(default=Path("results"))
    out: Path = Field(default=Path("paper/figures/compaction.pdf"))


def load_curve(results: Path, kind: str, compacted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation of causal retrieval gain over seeds, by gap."""
    curves = []
    for seed in SEEDS:
        run_dir = results / f"roomnav_transition_v1_{kind}_h128_s{seed}"
        if compacted:
            report = json.loads((run_dir / "compacted" / "probe.json").read_text())["report"]
        else:
            report = json.loads((run_dir / "probe_20000.json").read_text())["report"]
        curves.append([report[label]["gain_mean"] for label in GAP_LABELS])
    values = np.asarray(curves)
    return values.mean(axis=0), values.std(axis=0)


def main(settings: Settings) -> None:
    apply_style()
    # A two by two grid leaves room for every gap label at 8 points without rotation.
    plot = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.6), sharey=True)
    figure = plot[0]
    axes = plot[1].reshape(-1)
    positions = np.arange(len(GAP_LABELS))
    for index in range(len(ARMS)):
        axis = axes[index]
        arm = ARMS[index]
        kind = arm[0]
        raw = load_curve(settings.results, kind, compacted=False)
        compacted = load_curve(settings.results, kind, compacted=True)
        # The gray band marks gaps the 128-step raw window cannot contain.
        axis.axvspan(4.5, 6.5, color=BAND_COLOR, zorder=-2, linewidth=0)
        axis.axhline(0, color="#777777", linewidth=0.6, zorder=-1)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=-1)
        # Solid black reads the raw window and dashed gray the compacted context.
        for curve, marker, style, color in ((raw, "o", "-", "black"), (compacted, "^", "--", GRAY)):
            axis.plot(
                positions, curve[0], color=color, marker=marker, linestyle=style, markersize=3.5
            )
        axis.set_title(arm[1], pad=4)
        axis.set_xticks(positions, GAP_LABELS)
        axis.set_xlim(-0.35, 6.35)
        axis.set_ylim(-0.02, 0.75)
        axis.set_xlabel("Steps Since Original Visit")
        axis.set_yticks((0.0, 0.2, 0.4, 0.6))
    axes[0].set_ylabel("Context Retrieval Gain")
    axes[2].set_ylabel("Context Retrieval Gain")
    # One legend names the two contexts once for all four panels.
    handles = [
        Line2D([], [], color="black", marker="o", markersize=3.5, label="Raw Context Window"),
        Line2D(
            [],
            [],
            color=GRAY,
            marker="^",
            linestyle="--",
            markersize=3.5,
            label="Compact Every Step",
        ),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5, h_pad=1.0)
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
