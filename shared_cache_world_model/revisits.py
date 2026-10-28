"""Find every revisit in a recording: a step that enters a room the walk has been in before."""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import tyro
from pydantic import BaseModel, DirectoryPath, Field


class RevisitSettings(BaseModel):
    recording: DirectoryPath = Field(description="directory holding steps.npz from record")
    # Gap bins are powers of two, so each doubling of the gap gets one bucket in the figure.
    bins: tuple[int, ...] = Field(default=(1, 8, 16, 32, 64, 128, 256, 512))


class Revisit(BaseModel):
    """A step that enters a room again, and the last earlier step spent in that room."""

    step: int
    previous_step: int
    gap: int


def validate_gap_bins(bins: tuple[int, ...]) -> None:
    """Require at least one positive, strictly increasing gap interval."""
    if len(bins) < 2:
        raise ValueError(f"gap bins need at least two edges, got {bins}")
    if bins[0] < 1:
        raise ValueError(f"the first gap bin edge must be positive, got {bins[0]}")
    if any(high <= low for low, high in pairwise(bins)):
        raise ValueError(f"gap bin edges must be strictly increasing, got {bins}")


def find_revisits(episode: np.ndarray, room: np.ndarray) -> list[Revisit]:
    """Pair each step that enters a room with the last earlier step inside the same room."""
    last_seen: dict[tuple[int, int], int] = {}
    revisits: list[Revisit] = []
    for step in range(len(room)):
        place = (int(episode[step]), int(room[step]))
        entered = step == 0 or place != (int(episode[step - 1]), int(room[step - 1]))
        if entered and place in last_seen:
            previous_step = last_seen[place]
            revisits.append(
                Revisit(step=step, previous_step=previous_step, gap=step - previous_step)
            )
        last_seen[place] = step
    return revisits


def tile_disagreement(tiles: np.ndarray, revisits: list[Revisit]) -> np.ndarray:
    """Per revisit, the fraction of room tiles that differ from the earlier visit."""
    # tiles: (steps, 16, 16). The generator makes a room once and never changes it, so every
    # entry here is zero and a number above zero means the recording is wrong.
    return np.array(
        [np.mean(tiles[revisit.step] != tiles[revisit.previous_step]) for revisit in revisits]
    )


def gap_histogram(revisits: list[Revisit], bins: tuple[int, ...]) -> dict[str, int]:
    validate_gap_bins(bins)
    gaps = np.array([r.gap for r in revisits])
    counts = np.histogram(gaps, bins=list(bins))[0]
    labels = [f"{low}-{high - 1}" for low, high in pairwise(bins)]
    return dict(zip(labels, counts.tolist(), strict=True))


def main(settings: RevisitSettings) -> None:
    recording = np.load(settings.recording / "steps.npz")
    revisits = find_revisits(recording["episode"], recording["room"])
    disagreement = tile_disagreement(recording["tiles"], revisits)
    report = {
        "steps": len(recording["room"]),
        "revisits": len(revisits),
        "gap_histogram": gap_histogram(revisits, settings.bins),
        "exact_match_fraction": float(np.mean(disagreement == 0)) if revisits else None,
        "mean_tile_disagreement": float(disagreement.mean()) if revisits else None,
    }
    rows = [r.model_dump() for r in revisits]
    (settings.recording / "revisits.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(tyro.cli(RevisitSettings))
