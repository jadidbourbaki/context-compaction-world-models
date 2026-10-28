"""Build the two results figures: the same context length and the same KV cache budget."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import BAND_COLOR, COLUMN_WIDTH, GRID_COLOR, LINE_STYLES, apply_style
from pydantic import BaseModel, DirectoryPath, Field

SEEDS = (42, 43, 44)
GAP_LABELS = ("1-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511")


class Settings(BaseModel):
    """Paths used to construct the paper result figure."""

    results: DirectoryPath = Field(default=Path("results"))
    out_same_context: Path = Field(default=Path("paper/figures/retrieval_same_context.pdf"))
    out_same_budget: Path = Field(default=Path("paper/figures/retrieval_same_budget.pdf"))


def load_curve(results: Path, run_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one gain curve and return its mean and population standard deviation."""
    curves = []
    for seed in SEEDS:
        summary_path = results / f"{run_prefix}{seed}" / "summary.json"
        summary = json.loads(summary_path.read_text())
        curves.append([summary["probe"][label]["gain_mean"] for label in GAP_LABELS])
    values = np.asarray(curves)
    return values.mean(axis=0), values.std(axis=0)


def plot_curve(axis: plt.Axes, curve: tuple[np.ndarray, np.ndarray], label: str, kind: str) -> None:
    """Draw one mean curve over seeds in the design's grayscale style."""
    positions = np.arange(len(GAP_LABELS))
    axis.plot(positions, curve[0], label=label, markersize=3.5, **LINE_STYLES[kind])


def format_axis(axis: plt.Axes) -> None:
    """Apply the shared axis formatting."""
    positions = np.arange(len(GAP_LABELS))
    axis.axvspan(4.5, 6.5, color=BAND_COLOR, zorder=-2, linewidth=0)
    axis.axhline(0, color="#777777", linewidth=0.6, zorder=-1)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=-1)
    axis.set_xticks(positions, GAP_LABELS)
    # Thin end margins spread the seven bins, so the two widest labels at the right stay apart.
    axis.set_xlim(-0.15, 6.15)
    axis.set_ylim(-0.02, 0.75)
    axis.set_yticks((0.0, 0.2, 0.4, 0.6))
    axis.set_xlabel("Steps Since Original Visit")


def main(settings: Settings) -> None:
    apply_style()
    same_layered = load_curve(settings.results, "roomnav_transition_v1_per_layer_h128_s")
    same_ced_full = load_curve(settings.results, "roomnav_transition_v1_ced_full_h128_s")
    same_ced = load_curve(settings.results, "roomnav_transition_v1_ced_h128_s")
    shared = load_curve(settings.results, "roomnav_transition_v1_shared_cache_h128_s")
    bounded_layered = load_curve(settings.results, "roomnav_equal_memory_v1_per_layer_w16_h128_s")
    bounded_ced = load_curve(settings.results, "roomnav_equal_memory_v1_ced_g32_h128_s")
    bounded_ced_full = load_curve(
        settings.results, "roomnav_equal_memory_v1_ced_full_w16_g16_h128_s"
    )

    same_context = (
        (same_layered, "Decoder-only", "per_layer"),
        (same_ced_full, "CED", "ced_full"),
        (same_ced, "CED-SWA", "ced"),
        (shared, "DD", "shared_cache"),
    )
    # The bounded runs keep each design's style. The caption gives their windows.
    same_budget = (
        (bounded_layered, "Decoder-only", "per_layer"),
        (bounded_ced_full, "CED", "ced_full"),
        (bounded_ced, "CED-SWA", "ced"),
        (shared, "DD", "shared_cache"),
    )
    draw_figure(same_context, settings.out_same_context)
    draw_figure(same_budget, settings.out_same_budget)


def draw_figure(
    curves: tuple[tuple[tuple[np.ndarray, np.ndarray], str, str], ...], out: Path
) -> None:
    """Draw one comparison as a single-column figure with its own legend."""
    # The cropped figure gains the 0.02 inch save padding on each side, so it is drawn that much
    # narrower to print at full size in the column.
    plot = plt.subplots(figsize=(COLUMN_WIDTH - 0.04, 2.1))
    figure = plot[0]
    axis = plot[1]
    format_axis(axis)
    for curve in curves:
        plot_curve(axis, curve[0], curve[1], curve[2])
    axis.set_ylabel("Context Retrieval Gain")
    handles = axis.get_legend_handles_labels()
    figure.legend(
        *handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        columnspacing=0.6,
        handlelength=1.8,
        handletextpad=0.3,
    )
    # The saved figure is cropped to its contents, so the layout keeps no padding of its own and
    # the axis takes the whole column.
    figure.tight_layout(rect=(0, 0, 1, 0.9), pad=0.1)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
