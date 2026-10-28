"""Build the game world figure: the observer's room, its doors, and the rooms they lead to."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from figure_style import COLUMN_WIDTH, apply_style
from matplotlib import patheffects
from matplotlib.patches import Circle, FancyArrowPatch
from pydantic import BaseModel, Field

from shared_cache_world_model.world import (
    ROOM_COLS,
    ROOM_ROWS,
    SPRITE_SIZE,
    Doors,
    WorldSettings,
    add_character,
    generate_world,
    render_room,
    room_doors,
)

FIGURE_SIZE = COLUMN_WIDTH
ROOM_PIXELS = ROOM_ROWS * SPRITE_SIZE
GAP_PIXELS = SPRITE_SIZE
PITCH_PIXELS = ROOM_PIXELS + GAP_PIXELS
# Neighbours keep this share of their ink, so the eye stays on the observer's room.
NEIGHBOUR_INK = 0.35


class Settings(BaseModel):
    """Inputs for the game world figure."""

    # The evaluation recording's seed, so the world drawn is its first episode's world.
    seed: int = Field(default=43)
    room_row: int = Field(default=3)
    room_col: int = Field(default=3)
    out: Path = Field(default=Path("paper/figures/benchmark.pdf"))


class Exit(BaseModel):
    """One door of the observer's room and the direction it leads."""

    row: int
    col: int
    step_row: int
    step_col: int


def room_pixels(screen: np.ndarray) -> np.ndarray:
    """Render one room as sprites without the black gutter render_room draws around it."""
    bordered = render_room(screen)
    return bordered[SPRITE_SIZE:-SPRITE_SIZE, SPRITE_SIZE:-SPRITE_SIZE]


def exits(doors: Doors) -> list[Exit]:
    """The doors of the observer's room, each with the direction it leads."""
    found: list[Exit] = []
    if doors.north is not None:
        found.append(Exit(row=0, col=doors.north, step_row=-1, step_col=0))
    if doors.south is not None:
        found.append(Exit(row=ROOM_ROWS - 1, col=doors.south, step_row=1, step_col=0))
    if doors.west is not None:
        found.append(Exit(row=doors.west, col=0, step_row=0, step_col=-1))
    if doors.east is not None:
        found.append(Exit(row=doors.east, col=ROOM_COLS - 1, step_row=0, step_col=1))
    return found


def main(settings: Settings) -> None:
    apply_style()
    world_settings = WorldSettings(out=Path("unused"), seed=settings.seed)
    world = generate_world(np.random.default_rng(settings.seed), world_settings)
    canvas_pixels = 3 * ROOM_PIXELS + 2 * GAP_PIXELS
    # canvas: (800, 800, 3) uint8, the observer's room and its eight neighbours on white.
    canvas = np.full((canvas_pixels, canvas_pixels, 3), 255, dtype=np.uint8)
    for offset_row in range(3):
        for offset_col in range(3):
            row = settings.room_row + offset_row - 1
            col = settings.room_col + offset_col - 1
            screen = world.tiles[row, col]
            if offset_row == 1 and offset_col == 1:
                pixels = room_pixels(add_character(screen))
            else:
                faded = room_pixels(screen) * NEIGHBOUR_INK + 255 * (1 - NEIGHBOUR_INK)
                pixels = faded.astype(np.uint8)
            top = offset_row * PITCH_PIXELS
            left = offset_col * PITCH_PIXELS
            canvas[top : top + ROOM_PIXELS, left : left + ROOM_PIXELS] = pixels

    figure, axis = plt.subplots(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    axis.imshow(canvas, interpolation="none")
    # A white stroke under every mark keeps it legible over the sprites.
    halo = [patheffects.withStroke(linewidth=2.4, foreground="white")]
    # An arrow leaves each door and ends inside the neighbour it leads to.
    reach = GAP_PIXELS + 3 * SPRITE_SIZE
    doors = room_doors(world_settings, world, settings.room_row, settings.room_col)
    for door in exits(doors):
        door_x = PITCH_PIXELS + SPRITE_SIZE * door.col + SPRITE_SIZE / 2 - 0.5
        door_y = PITCH_PIXELS + SPRITE_SIZE * door.row + SPRITE_SIZE / 2 - 0.5
        start_x = door_x + door.step_col * SPRITE_SIZE * 0.7
        start_y = door_y + door.step_row * SPRITE_SIZE * 0.7
        end_x = door_x + door.step_col * reach
        end_y = door_y + door.step_row * reach
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle="-|>,head_length=0.55,head_width=0.28",
            mutation_scale=9,
            color="black",
            linewidth=0.9,
        )
        arrow.set_path_effects(halo)
        axis.add_patch(arrow)
    observer_x = PITCH_PIXELS + SPRITE_SIZE * (ROOM_COLS // 2) + SPRITE_SIZE / 2 - 0.5
    observer_y = PITCH_PIXELS + SPRITE_SIZE * (ROOM_ROWS // 2) + SPRITE_SIZE / 2 - 0.5
    ring = Circle(
        (observer_x, observer_y),
        radius=SPRITE_SIZE * 0.85,
        fill=False,
        edgecolor="black",
        linewidth=0.9,
    )
    ring.set_path_effects(halo)
    axis.add_patch(ring)
    label = axis.annotate(
        "Observer",
        xy=(observer_x - SPRITE_SIZE * 0.6, observer_y - SPRITE_SIZE * 0.6),
        xytext=(observer_x - SPRITE_SIZE * 2.5, observer_y - SPRITE_SIZE * 2.5),
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.6,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "black",
            "linewidth": 0.7,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    label.arrow_patch.set_path_effects(halo)
    # The observer's room and a strip of each neighbour, enough to show where a door leads.
    margin = GAP_PIXELS + 3.5 * SPRITE_SIZE
    axis.set_xlim(PITCH_PIXELS - margin, PITCH_PIXELS + ROOM_PIXELS + margin)
    axis.set_ylim(PITCH_PIXELS + ROOM_PIXELS + margin, PITCH_PIXELS - margin)
    axis.axis("off")
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out)


if __name__ == "__main__":
    main(tyro.cli(Settings))
