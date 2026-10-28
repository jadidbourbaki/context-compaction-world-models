"""Build the first figure: the context each cache fits, its gain while dreaming, and one revisit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from figure_style import ARM_FILLS, ARM_HATCHES, ARMS, TEXT_WIDTH, apply_style
from matplotlib.patches import Rectangle
from pydantic import BaseModel, DirectoryPath, Field

from shared_cache_world_model.agent import load_model
from shared_cache_world_model.data import (
    Recording,
    SplitSettings,
    load_recording,
    model_frames,
)
from shared_cache_world_model.model import WorldModel
from shared_cache_world_model.probe import ContextMode, probe_context
from shared_cache_world_model.revisits import Revisit
from shared_cache_world_model.world import ROOM_COLS, ROOM_ROWS, add_character, render_room

HISTORY = 128
SEEDS = (42, 43, 44)
PIXEL_REPEAT = 4
EXAMPLE_COLOR = ARMS[0][2]
TILE_LABELS = ("Original Visit", "Revisit (Raw Window)", "Revisit (Context Compaction)")


class Settings(BaseModel):
    """Inputs for the first figure."""

    results: DirectoryPath = Field(default=Path("results"))
    weights: DirectoryPath = Field(default=Path("weights"))
    recording: DirectoryPath = Field(default=Path("data/roomnav_eval"))
    # The revisit at step 1333 enters a room last seen 262 steps earlier, about twice the window.
    revisit_step: int = Field(default=1333)
    out: Path = Field(default=Path("paper/figures/teaser.pdf"))


def predicted_room(
    model: WorldModel, recording: Recording, revisit: Revisit, context_mode: ContextMode
) -> np.ndarray:
    """The entering room as the model predicts it from one context, as tile ids."""
    context = probe_context(recording, revisit, HISTORY, context_mode)
    # frames: (1, steps, 16, 16) int64 and actions: (1, steps), a batch of one context.
    frames = model_frames(context.frames)[None].long()
    actions = context.actions[None]
    with torch.no_grad():
        # logits: (1, steps, 8, 256). The step before the entering step predicts its tiles.
        logits = model(frames, actions)
    steps = int(frames.shape[1])
    predicted = logits[0, steps - 2].argmax(dim=0)
    return predicted.reshape(ROOM_ROWS, ROOM_COLS).numpy()


def draw_room(axis: plt.Axes, screen: np.ndarray, outline: str) -> None:
    """Draw one room with sharp sprite pixels and a coloured outline."""
    # Repeating every pixel four times keeps the one-pixel sprite lines sharp at print size.
    pixels = render_room(screen).repeat(PIXEL_REPEAT, axis=0).repeat(PIXEL_REPEAT, axis=1)
    axis.imshow(pixels, interpolation="none")
    axis.add_patch(
        Rectangle(
            (0, 0), 1, 1, transform=axis.transAxes, fill=False, edgecolor=outline, linewidth=1.2
        )
    )
    axis.axis("off")


def draw_example(axes: list[plt.Axes], settings: Settings) -> None:
    """The earlier visit and the per-layer model's prediction through each context."""
    device = torch.device("cpu")
    checkpoint = settings.weights / "roomnav_transition_v1_per_layer_h128_s42" / "weights.pt"
    model = load_model(checkpoint, device)[0]
    recording = load_recording(SplitSettings(recording=settings.recording))
    rooms = recording.rooms.numpy()
    end = settings.revisit_step
    previous_visit = int(np.flatnonzero(rooms[:end] == rooms[end])[-1])
    revisit = Revisit(step=end, previous_step=previous_visit, gap=end - previous_visit)
    earlier = add_character(recording.tiles[previous_visit].numpy())
    draw_room(axes[0], earlier, EXAMPLE_COLOR)
    context_modes: tuple[ContextMode, ...] = ("raw", "compacted")
    for index in range(len(context_modes)):
        context_mode = context_modes[index]
        screen = predicted_room(model, recording, revisit, context_mode)
        outline = EXAMPLE_COLOR if context_mode == "compacted" else "#999999"
        draw_room(axes[index + 1], screen, outline)


def history_in_budget(cache_bytes: dict[str, int], budget: int) -> int:
    """The longest history whose cache fits in ``budget`` bytes."""
    # The cache grows linearly in the history: a fixed part for the windowed blocks plus a
    # per-step part, read off the measured sizes at 64 and 128 steps.
    per_step = (cache_bytes["128"] - cache_bytes["64"]) / 64
    fixed = cache_bytes["128"] - 128 * per_step
    return int((budget - fixed) // per_step)


def draw_history(axis: plt.Axes, results: Path) -> None:
    """How many steps of history each design's cache fits in the per-layer cache at 128 steps."""
    systems = json.loads((results / "systems" / "summary.json").read_text())
    cache_bytes = {design["kind"]: design["cache_bytes"] for design in systems["designs"]}
    budget = cache_bytes["per_layer"]["128"]
    for index in range(len(ARMS)):
        kind = ARMS[index][0]
        steps = history_in_budget(cache_bytes[kind], budget)
        axis.bar(
            index,
            steps,
            width=0.65,
            color=ARM_FILLS[index],
            hatch=ARM_HATCHES[index],
            edgecolor="black",
            linewidth=0.6,
            label=ARMS[index][1],
        )
    axis.set_xticks([])
    axis.set_ylim(0, 1100)
    axis.set_yticks([0, 256, 512, 768, 1024])
    axis.set_ylabel("Context Size (Steps)")


def draw_dreaming(axis: plt.Axes, results: Path) -> None:
    """Each design's context retrieval gain while dreaming with context compaction."""
    for index in range(len(ARMS)):
        kind = ARMS[index][0]
        gains = []
        for seed in SEEDS:
            run_dir = results / f"roomnav_transition_v1_{kind}_h128_s{seed}"
            # The revisit that closes each 128-step loop, with compact every step and dreaming.
            path = run_dir / "agent" / "loop128_compacted_dreaming.json"
            gains.append(json.loads(path.read_text())["loop_gap"]["retrieval_gain"])
        axis.bar(
            index,
            float(np.mean(gains)),
            width=0.65,
            color=ARM_FILLS[index],
            hatch=ARM_HATCHES[index],
            edgecolor="black",
            linewidth=0.6,
        )
    axis.set_xticks([])
    axis.set_ylim(0, 0.5)
    axis.set_yticks([0.0, 0.2, 0.4])
    axis.set_ylabel("Context Retrieval Gain")


def main(settings: Settings) -> None:
    apply_style()
    # Every label in this figure, from tile labels to tick labels, prints at one size.
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure = plt.figure(figsize=(TEXT_WIDTH, 1.95))
    # The bar panels and the rooms get their own grids, so each bar panel has room for its
    # y label and the three rooms sit close together.
    outer = figure.add_gridspec(1, 2, width_ratios=(2.0, 3.0), wspace=0.1, left=0.06, right=0.99)
    bars = outer[0, 0].subgridspec(1, 2, wspace=0.65)
    rooms = outer[0, 1].subgridspec(1, 3, wspace=0.06)
    history_axis = figure.add_subplot(bars[0, 0])
    dreaming_axis = figure.add_subplot(bars[0, 1])
    example_axes = [figure.add_subplot(rooms[0, column]) for column in range(3)]
    draw_example(example_axes, settings)
    draw_history(history_axis, settings.results)
    draw_dreaming(dreaming_axis, settings.results)
    # The bar panels take the rooms' box, so their x axes sit level with the rooms' bottom edge.
    example_axes[0].apply_aspect()
    room_box = example_axes[0].get_position()
    for axis in (history_axis, dreaming_axis):
        box = axis.get_position()
        axis.set_position((box.x0, room_box.y0, box.width, room_box.height))
    # One legend above the two bar panels names the four designs.
    bars_center = (history_axis.get_position().x0 + dreaming_axis.get_position().x1) / 2
    handles = history_axis.get_legend_handles_labels()
    figure.legend(
        *handles,
        loc="lower center",
        bbox_to_anchor=(bars_center, room_box.y1 + 0.01),
        ncol=len(ARMS),
        frameon=False,
        columnspacing=0.6,
        handlelength=1.0,
        handletextpad=0.3,
    )
    # Each tile label sits under its room, and each part of the figure is named one line below.
    label_top = room_box.y0 - 0.03
    layer_top = label_top - 0.1
    for index in range(len(example_axes)):
        box = example_axes[index].get_position()
        center = (box.x0 + box.x1) / 2
        figure.text(center, label_top, TILE_LABELS[index], ha="center", va="top")
    harness_center = (example_axes[0].get_position().x0 + example_axes[-1].get_position().x1) / 2
    figure.text(harness_center, layer_top, "Agent Harness Layer", ha="center", va="top")
    for axis, name in ((history_axis, "Model Layer"), (dreaming_axis, "Dreaming")):
        box = axis.get_position()
        # The bar panels carry no tick labels, so their names take the first label line.
        figure.text((box.x0 + box.x1) / 2, label_top, name, ha="center", va="top")
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main(tyro.cli(Settings))
