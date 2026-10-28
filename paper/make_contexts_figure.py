"""Build the appendix figure: the gain beyond the window of each context the harness sends."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import ARMS, COLUMN_WIDTH, apply_style, design_legend, draw_design_bars
from make_interaction_figure import BEYOND_WINDOW_BINS, per_seed
from pydantic import BaseModel, DirectoryPath, Field

# The contexts in the order of how much of the episode they keep.
CONTEXTS = (
    ("raw", "Raw Context\nWindow"),
    ("truncated", "Truncated"),
    ("compacted", "Compact\nEvery Step"),
    ("compacted_when_full", "Compact\nWhen Full"),
)


class Settings(BaseModel):
    """Paths used to construct the appendix context figure."""

    results: DirectoryPath = Field(default=Path("results"))
    out: Path = Field(default=Path("paper/figures/contexts.pdf"))


def main(settings: Settings) -> None:
    apply_style()
    plot = plt.subplots(figsize=(COLUMN_WIDTH, 1.9))
    figure = plot[0]
    axis = plot[1]
    # means and spreads: (contexts, designs), the mean and population std over seeds.
    means = np.zeros((len(CONTEXTS), len(ARMS)))
    spreads = np.zeros((len(CONTEXTS), len(ARMS)))
    for context_index in range(len(CONTEXTS)):
        for arm_index in range(len(ARMS)):
            values = per_seed(
                settings.results,
                ARMS[arm_index][0],
                CONTEXTS[context_index][0],
                BEYOND_WINDOW_BINS,
                "gain_mean",
            )
            means[context_index, arm_index] = values.mean()
            spreads[context_index, arm_index] = values.std()
    draw_design_bars(axis, tuple(context[1] for context in CONTEXTS), means, spreads)
    axis.set_ylim(0, 0.7)
    axis.set_ylabel("Context Retrieval Gain")
    design_legend(figure, axis)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
