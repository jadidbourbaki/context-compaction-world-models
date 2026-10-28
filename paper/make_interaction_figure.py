"""Build the agent harness figure: the reach of every design under three contexts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import ARMS, COLUMN_WIDTH, apply_style, design_legend, draw_design_bars
from pydantic import BaseModel, DirectoryPath, Field

SEEDS = (42, 43, 44)
# The three contexts of the figure, from the least to the most of the episode they keep.
CONTEXTS = (
    ("raw", "Raw Context\nWindow"),
    ("compacted", "Compact\nEvery Step"),
    ("compacted_when_full", "Compact\nWhen Full"),
)
IN_WINDOW_BINS = ("1-7", "8-15", "16-31", "32-63", "64-127")
BEYOND_WINDOW_BINS = ("128-255", "256-511")


class Settings(BaseModel):
    """Paths used to construct the agent harness figure."""

    results: DirectoryPath = Field(default=Path("results"))
    out: Path = Field(default=Path("paper/figures/interaction.pdf"))


def load_report(results: Path, kind: str, seed: int, context: str) -> dict[str, dict[str, float]]:
    """The per-bin probe report of one model read through one context."""
    run_dir = results / f"roomnav_transition_v1_{kind}_h128_s{seed}"
    # The raw-window probe is the one written at the end of training.
    path = run_dir / "probe_20000.json" if context == "raw" else run_dir / context / "probe.json"
    return json.loads(path.read_text())["report"]


def per_seed(
    results: Path, kind: str, context: str, bins: tuple[str, ...], field: str
) -> np.ndarray:
    """One probe field averaged over the bins, for each seed: (seeds,)."""
    values = []
    for seed in SEEDS:
        report = load_report(results, kind, seed, context)
        values.append(np.mean([report[label][field] for label in bins]))
    return np.asarray(values)


def draw_reach(axis: plt.Axes, results: Path) -> None:
    """Grouped bars of the gain beyond 128 steps: one group per context, one bar per design."""
    # means and spreads: (contexts, designs), the mean and population std over seeds.
    means = np.zeros((len(CONTEXTS), len(ARMS)))
    spreads = np.zeros((len(CONTEXTS), len(ARMS)))
    for context_index in range(len(CONTEXTS)):
        for arm_index in range(len(ARMS)):
            values = per_seed(
                results,
                ARMS[arm_index][0],
                CONTEXTS[context_index][0],
                BEYOND_WINDOW_BINS,
                "gain_mean",
            )
            means[context_index, arm_index] = values.mean()
            spreads[context_index, arm_index] = values.std()
    draw_design_bars(axis, tuple(context[1] for context in CONTEXTS), means, spreads)
    axis.set_ylim(0, 0.7)
    axis.set_yticks((0.0, 0.2, 0.4, 0.6))
    axis.set_ylabel("Context Retrieval Gain")


def main(settings: Settings) -> None:
    apply_style()
    plot = plt.subplots(figsize=(COLUMN_WIDTH, 1.75))
    figure = plot[0]
    axis = plot[1]
    draw_reach(axis, settings.results)
    design_legend(figure, axis)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
