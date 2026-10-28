"""Build the latency figure: the mean agent latency per step, split into model and harness time."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import (
    ARMS,
    COLUMN_WIDTH,
    DESIGN_BAR_WIDTH,
    apply_style,
    design_legend,
    draw_design_bars,
)
from matplotlib.patches import Patch
from pydantic import BaseModel, Field, FilePath

# The policies in the order of the other agent harness figures.
POLICIES = (
    ("raw", "Raw Context\nWindow"),
    ("compacted", "Compact\nEvery Step"),
    ("compacted_when_full", "Compact\nWhen Full"),
)
HARNESS_FILL = "#333333"


class Settings(BaseModel):
    """Paths used to construct the latency figure."""

    summary: FilePath = Field(default=Path("results/session/summary.json"))
    out: Path = Field(default=Path("paper/figures/latency.pdf"))


def main(settings: Settings) -> None:
    apply_style()
    summary = json.loads(settings.summary.read_text())
    step_ms = {design["kind"]: design["step_ms"] for design in summary["designs"]}
    # harness_ms[policy][step]: the harness time of each step, the same for every design.
    harness_ms = summary["plan"]["harness_ms"]
    plot = plt.subplots(figsize=(COLUMN_WIDTH, 1.75))
    figure = plot[0]
    axis = plot[1]
    # inference and harness: (policies, designs), the mean time per step over the whole session
    # inside the world model and inside the agent harness.
    inference = np.zeros((len(POLICIES), len(ARMS)))
    harness = np.zeros((len(POLICIES), len(ARMS)))
    for policy_index in range(len(POLICIES)):
        policy = POLICIES[policy_index][0]
        planning = np.asarray(harness_ms[policy])
        for arm_index in range(len(ARMS)):
            agent = np.asarray(step_ms[ARMS[arm_index][0]][policy])
            inference[policy_index, arm_index] = np.mean(agent - planning)
            harness[policy_index, arm_index] = np.mean(planning)
    draw_design_bars(axis, tuple(policy[1] for policy in POLICIES), inference)
    # The harness latency sits on top of each bar as a dark segment, darker than every design
    # fill, so the full height of a bar is the agent latency.
    for policy_index in range(len(POLICIES)):
        for arm_index in range(len(ARMS)):
            axis.bar(
                policy_index + (arm_index - 1.5) * DESIGN_BAR_WIDTH,
                harness[policy_index, arm_index],
                bottom=inference[policy_index, arm_index],
                width=DESIGN_BAR_WIDTH * 0.9,
                color=HARNESS_FILL,
                edgecolor="black",
                linewidth=0.6,
            )
    axis.set_ylim(0, 20)
    axis.set_yticks((0, 10, 20))
    axis.set_ylabel("Agent Latency (ms)")
    design_legend(figure, axis)
    # The design legend fills the row above the plot, so the harness segment is named in the
    # empty space above the compact when full bars.
    harness_handle = Patch(facecolor=HARNESS_FILL, edgecolor="black", linewidth=0.6)
    axis.legend(
        [harness_handle],
        ["Harness Latency"],
        loc="upper right",
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
