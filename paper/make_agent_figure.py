"""Build the closed-loop agent figure: context retrieval gain at each loop's closing revisit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
    ARMS,
    GRID_COLOR,
    TEXT_WIDTH,
    apply_style,
)
from matplotlib.lines import Line2D
from pydantic import BaseModel, DirectoryPath, Field

SEEDS = (42, 43, 44)
LOOP_LENGTHS = (8, 32, 128)
BAR_WIDTH = 0.36
# Each condition pairs a context with dreaming on or off. Solid black lines read the raw context
# window and dashed gray lines the compacted context. Filled markers observe revisits and open
# markers dream them.
CONDITIONS = (
    ("raw", "no-dreaming", "Raw Context Window, No Dreaming", "-", "o", True, "black"),
    ("raw", "dreaming", "Raw Context Window, Dreaming", "-", "o", False, "black"),
    (
        "compacted",
        "no-dreaming",
        "Compact Every Step, No Dreaming",
        "--",
        "^",
        True,
        "#8c8c8c",
    ),
    ("compacted", "dreaming", "Compact Every Step, Dreaming", "--", "^", False, "#8c8c8c"),
)


class Settings(BaseModel):
    """Paths used to construct the agent figure."""

    results: DirectoryPath = Field(default=Path("results"))
    sweep_out: Path = Field(default=Path("paper/figures/agent_sweep.pdf"))


def load_curve(
    results: Path, kind: str, context: str, dreaming: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation over seeds of closing-revisit gain, by loop."""
    values = load_seeds(results, kind, context, dreaming)
    return values.mean(axis=0), values.std(axis=0)


def load_seeds(results: Path, kind: str, context: str, dreaming: str) -> np.ndarray:
    """Closing-revisit context retrieval gain: (seeds, loop lengths)."""
    curves = []
    for seed in SEEDS:
        agent_dir = results / f"roomnav_transition_v1_{kind}_h128_s{seed}" / "agent"
        row = []
        for loop_steps in LOOP_LENGTHS:
            path = agent_dir / f"loop{loop_steps}_{context}_{dreaming}.json"
            report = json.loads(path.read_text())
            row.append(report["loop_gap"]["retrieval_gain"])
        curves.append(row)
    return np.asarray(curves)


def draw_sweep(settings: Settings) -> None:
    """Four panels: closing-revisit gain against loop length for every condition."""
    plot = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.0), sharey=True)
    figure = plot[0]
    axes = plot[1]
    positions = np.arange(len(LOOP_LENGTHS))
    for index in range(len(ARMS)):
        axis = axes[index]
        arm = ARMS[index]
        kind = arm[0]
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=-1)
        for condition in CONDITIONS:
            curve = load_curve(settings.results, kind, condition[0], condition[1])
            axis.errorbar(
                positions,
                curve[0],
                yerr=curve[1],
                color=condition[6],
                linestyle=condition[3],
                marker=condition[4],
                markersize=3.5,
                markerfacecolor=condition[6] if condition[5] else "white",
            )
        axis.set_title(arm[1], pad=4)
        axis.set_xticks(positions, [str(length) for length in LOOP_LENGTHS])
        axis.set_xlim(-0.3, len(LOOP_LENGTHS) - 0.7)
        axis.set_ylim(-0.02, 0.75)
        axis.set_yticks((0.0, 0.2, 0.4, 0.6))
        axis.set_xlabel("Loop Length (Steps)")
    axes[0].set_ylabel("Context Retrieval Gain")
    # The legend names each factor once. Line style and colour give the context, and the
    # marker fill gives dreaming, so four short entries fit on one row.
    handles = [
        Line2D([], [], color="black", linestyle="-", label="Raw Context Window"),
        Line2D([], [], color="#8c8c8c", linestyle="--", label="Compact Every Step"),
        Line2D(
            [],
            [],
            color="black",
            linestyle="",
            marker="o",
            markersize=3.5,
            label="No Dreaming",
        ),
        Line2D(
            [],
            [],
            color="black",
            linestyle="",
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            label="Dreaming",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        columnspacing=0.5,
        handlelength=1.3,
        handletextpad=0.25,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
    settings.sweep_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.sweep_out)
    plt.close(figure)


def main(settings: Settings) -> None:
    apply_style()
    draw_sweep(settings)


if __name__ == "__main__":
    main(tyro.cli(Settings))
