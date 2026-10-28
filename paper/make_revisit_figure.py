"""Build the revisit figure: loop-closing gain with observed and with predicted revisits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import ARMS, COLUMN_WIDTH, apply_style, design_legend, draw_design_bars
from pydantic import BaseModel, DirectoryPath, Field

SEEDS = (42, 43, 44)
LOOP_STEPS = 128
# One panel per feedback condition. The agent reports name predicted revisits "dreaming".
PANELS = (("no-dreaming", "No Dreaming"), ("dreaming", "Dreaming"))
CONTEXTS = (("raw", "Raw Context\nWindow"), ("compacted", "Compact\nEvery Step"))


class Settings(BaseModel):
    """Paths used to construct the revisit figure."""

    results: DirectoryPath = Field(default=Path("results"))
    out: Path = Field(default=Path("paper/figures/revisits.pdf"))


def per_seed(results: Path, kind: str, context: str, feedback: str) -> np.ndarray:
    """The context retrieval gain at the revisit closing each 128-step loop, per seed: (seeds,)."""
    values = []
    for seed in SEEDS:
        run_dir = results / f"roomnav_transition_v1_{kind}_h128_s{seed}"
        path = run_dir / "agent" / f"loop{LOOP_STEPS}_{context}_{feedback}.json"
        values.append(json.loads(path.read_text())["loop_gap"]["retrieval_gain"])
    return np.asarray(values)


def main(settings: Settings) -> None:
    apply_style()
    plot = plt.subplots(1, len(PANELS), figsize=(COLUMN_WIDTH, 1.9), sharey=True)
    figure = plot[0]
    axes = plot[1]
    for panel_index in range(len(PANELS)):
        feedback = PANELS[panel_index][0]
        # means: (contexts, designs), the mean gain over seeds.
        means = np.zeros((len(CONTEXTS), len(ARMS)))
        for context_index in range(len(CONTEXTS)):
            for arm_index in range(len(ARMS)):
                values = per_seed(
                    settings.results, ARMS[arm_index][0], CONTEXTS[context_index][0], feedback
                )
                means[context_index, arm_index] = values.mean()
        axis = axes[panel_index]
        draw_design_bars(axis, tuple(context[1] for context in CONTEXTS), means)
        axis.set_title(PANELS[panel_index][1], pad=4)
        axis.set_ylim(0, 0.6)
        axis.set_yticks((0.0, 0.2, 0.4, 0.6))
    axes[0].set_ylabel("Context Retrieval Gain")
    design_legend(figure, axes[0])
    figure.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.0)
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
