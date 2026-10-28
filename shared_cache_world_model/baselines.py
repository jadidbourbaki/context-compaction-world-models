"""Measure the reference points the probe is read against, on one recording."""

from __future__ import annotations

import json
import zlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tyro
from pydantic import BaseModel, DirectoryPath, Field

from shared_cache_world_model.data import (
    Recording,
    SplitSettings,
    load_recording,
    recording_identity,
)
from shared_cache_world_model.train import current_commit
from shared_cache_world_model.world import INTERIOR, TILE_VOCAB


class BaselineSettings(BaseModel):
    recording: DirectoryPath = Field(description="directory holding steps.npz from the game")
    out: Path | None = Field(default=None, description="where to write the table as JSON")


def most_common_tiles(tiles: np.ndarray) -> np.ndarray:
    """The tile id that appears most often at each position: (16, 16) uint8."""
    # counts: (8, 16, 16) int64, one plane per tile id, so the mode is an argmax over 8 planes.
    counts = np.zeros((TILE_VOCAB, *tiles.shape[1:]), dtype=np.int64)
    for tile in range(TILE_VOCAB):
        counts[tile] = (tiles == tile).sum(axis=0)
    return counts.argmax(axis=0).astype(np.uint8)


def bits_per_room(tiles: np.ndarray, rooms: np.ndarray) -> float:
    """How much a room costs to store, which sets how much a model must hold to memorize them."""
    # One frame per distinct room, compressed together, so repeated structure across rooms is
    # counted once and the figure is an upper bound on what the model would have to learn.
    first_frames: dict[int, np.ndarray] = {}
    for step in range(len(rooms)):
        first_frames.setdefault(int(rooms[step]), tiles[step])
    distinct = np.stack(list(first_frames.values()), axis=0)
    compressed = zlib.compress(distinct.tobytes(), 9)
    return len(compressed) * 8 / len(first_frames)


def measure(recording: Recording) -> dict[str, float]:
    """The accuracy of three ways of guessing every next room."""
    tiles = recording.tiles.numpy()
    rooms = recording.rooms.numpy()
    common = most_common_tiles(tiles)
    # The probe scores the same 196 cells inside the wall border.
    scored = INTERIOR
    # last_visit holds the most recent step spent in each room, which is the frame to copy.
    last_visit: dict[int, int] = {int(rooms[0]): 0}
    frequency: list[float] = []
    previous_frame: list[float] = []
    last_visit_frame: list[float] = []
    for step in range(1, len(rooms)):
        room = int(rooms[step])
        if recording.offset_in_episode(step) != 0:
            truth = tiles[step]
            frequency.append(float((common[scored] == truth[scored]).mean()))
            previous_frame.append(float((tiles[step - 1][scored] == truth[scored]).mean()))
            if room in last_visit:
                seen = tiles[last_visit[room]]
                last_visit_frame.append(float((seen[scored] == truth[scored]).mean()))
        last_visit[room] = step
    return {
        "rooms": float(len(last_visit)),
        "transition_frames": float(len(frequency)),
        "revisits": float(len(last_visit_frame)),
        "most_common_tile": float(np.mean(frequency)),
        "previous_frame": float(np.mean(previous_frame)),
        "last_visit_frame": float(np.mean(last_visit_frame)),
        "bits_per_room": bits_per_room(tiles, rooms),
    }


def main(settings: BaselineSettings) -> None:
    recording = load_recording(SplitSettings(recording=settings.recording))
    table = measure(recording)
    rounded = {key: round(value, 4) for key, value in table.items()}
    rendered = json.dumps(rounded, indent=2)
    if settings.out is not None:
        summary = {
            "baselines": rounded,
            "recording": str(settings.recording),
            "recording_sha256": recording_identity(settings.recording).sha256,
            "commit": current_commit(),
            "versions": {"numpy": version("numpy")},
        }
        settings.out.parent.mkdir(parents=True, exist_ok=True)
        settings.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(rendered)


if __name__ == "__main__":
    main(tyro.cli(BaselineSettings))
