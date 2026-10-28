from __future__ import annotations

import numpy as np
import pytest

from shared_cache_world_model.revisits import (
    Revisit,
    find_revisits,
    gap_histogram,
    tile_disagreement,
    validate_gap_bins,
)


def test_find_revisits_pairs_each_room_entry_with_the_last_stay() -> None:
    # Rooms A, A, B, A, A, C, A: the walk re-enters A at step 3 and at step 6.
    episode = np.zeros(7, dtype=int)
    room = np.array([1, 1, 2, 1, 1, 3, 1])
    assert find_revisits(episode, room) == [
        Revisit(step=3, previous_step=1, gap=2),
        Revisit(step=6, previous_step=4, gap=2),
    ]


def test_find_revisits_separates_maps_with_the_same_room_number() -> None:
    episode = np.array([0, 16, 0])
    room = np.array([7, 7, 7])
    assert find_revisits(episode, room) == [Revisit(step=2, previous_step=0, gap=2)]


def test_tile_disagreement_counts_changed_tiles() -> None:
    tiles = np.zeros((2, 16, 20), dtype=np.uint16)
    tiles[1, 0, :4] = 7
    revisits = [Revisit(step=1, previous_step=0, gap=1)]
    assert tile_disagreement(tiles, revisits).tolist() == pytest.approx([4 / 320])


def test_gap_histogram_labels_power_of_two_bins() -> None:
    revisits = [Revisit(step=g, previous_step=0, gap=g) for g in (1, 2, 3, 5, 9)]
    assert gap_histogram(revisits, (1, 2, 4, 8, 16)) == {"1-1": 1, "2-3": 2, "4-7": 1, "8-15": 1}


def test_gap_bins_must_be_positive_and_increasing() -> None:
    with pytest.raises(ValueError, match="at least two"):
        validate_gap_bins((1,))
    with pytest.raises(ValueError, match="positive"):
        validate_gap_bins((0, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_gap_bins((1, 8, 8))
