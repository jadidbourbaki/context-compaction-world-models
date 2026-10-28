from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from shared_cache_world_model.compaction import (
    compact,
    deduplicate,
    plan_walk,
    positions_from_actions,
)
from shared_cache_world_model.world import ACTIONS, MOVES, WorldSettings, record

RIGHT = ACTIONS.index("right")
LEFT = ACTIONS.index("left")


def grid_positions(rooms: np.ndarray, rooms_across: int) -> list[tuple[int, int]]:
    return [(int(room // rooms_across), int(room % rooms_across)) for room in rooms]


@given(
    world_seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps_per_episode=st.integers(min_value=2, max_value=40),
    rooms_across=st.integers(min_value=2, max_value=4),
    rooms_down=st.integers(min_value=2, max_value=4),
)
@hypothesis_settings(max_examples=20, deadline=None)
def test_the_walk_covers_every_room_once_planned_from_actions_alone(
    world_seed: int, steps_per_episode: int, rooms_across: int, rooms_down: int
) -> None:
    # Small grids force many revisits. The planner sees only actions. The checks below use the
    # recorded room ids it never saw and a grid graph it never built.
    settings = WorldSettings(
        out=Path("unused"),
        episodes=2,
        steps_per_episode=steps_per_episode,
        rooms_across=rooms_across,
        rooms_down=rooms_down,
        seed=world_seed,
        video_steps=0,
    )
    steps = record(settings)
    grid = nx.grid_2d_graph(rooms_down, rooms_across)
    for episode in range(settings.episodes):
        start = episode * steps_per_episode
        for offset in range(steps_per_episode):
            end = start + offset
            actions = steps.actions[start : end + 1]
            walk = plan_walk(actions)
            source_rooms = steps.room[start + walk.source_steps]
            positions = grid_positions(source_rooms, rooms_across)
            # A walk of adjacent rooms that ends where the transcript ends.
            for index in range(len(positions) - 1):
                assert grid.has_edge(positions[index], positions[index + 1])
            assert walk.source_steps[-1] == offset
            # Every room visited so far appears, and no other.
            visited_rooms = steps.room[start : end + 1]
            assert set(source_rooms.tolist()) == set(visited_rooms.tolist())
            assert len(visited_rooms) >= len(set(visited_rooms.tolist()))
            assert len(walk.source_steps) <= 2 * len(set(visited_rooms.tolist()))
            # Each room's frame comes from the latest step that stood in it.
            for source in walk.source_steps:
                same_room = np.nonzero(visited_rooms == steps.room[start + source])[0]
                assert source == same_room.max()
            # The written actions move along the walk, and the last is the transcript's own.
            for index in range(len(positions) - 1):
                move = MOVES[int(walk.actions[index])]
                assert (positions[index][0] + move[0], positions[index][1] + move[1]) == positions[
                    index + 1
                ]
            assert walk.actions[-1] == actions[-1]


@given(
    world_seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps_per_episode=st.integers(min_value=2, max_value=60),
)
@hypothesis_settings(max_examples=10, deadline=None)
def test_dead_reckoning_matches_the_recorded_rooms(world_seed: int, steps_per_episode: int) -> None:
    settings = WorldSettings(
        out=Path("unused"), episodes=1, steps_per_episode=steps_per_episode, seed=world_seed
    )
    steps = record(settings)
    positions = positions_from_actions(steps.actions)
    recorded = np.array(grid_positions(steps.room, settings.rooms_across))
    # Positions are relative to the first room, so compare displacements from it.
    assert np.array_equal(positions - positions[0], recorded - recorded[0])


def test_a_single_step_is_its_own_walk() -> None:
    walk = plan_walk(np.array([RIGHT], dtype=np.uint8))
    assert np.array_equal(walk.source_steps, np.array([0]))
    assert np.array_equal(walk.actions, np.array([RIGHT]))


def test_an_oscillation_between_two_rooms_compacts_to_two_steps() -> None:
    actions = np.array([RIGHT, LEFT] * 5, dtype=np.uint8)
    walk = plan_walk(actions)
    # Two rooms, so two steps, ending at the current room with its newest frame.
    assert len(walk.source_steps) == 2
    assert walk.source_steps[-1] == 9
    assert walk.actions[-1] == actions[-1]


def test_compact_gathers_each_rooms_newest_frame() -> None:
    actions = torch.tensor([RIGHT, RIGHT, LEFT, LEFT])
    tiles = torch.arange(4, dtype=torch.uint8)[:, None, None].expand(4, 16, 16)
    compacted = compact(actions, tiles)
    walk = plan_walk(actions.numpy())
    assert torch.equal(compacted.frames, tiles[torch.from_numpy(walk.source_steps)])
    assert torch.equal(compacted.actions, torch.from_numpy(walk.actions))


def test_deduplicate_keeps_each_rooms_newest_step_in_visit_order() -> None:
    # Positions 0, 1, 2, 1, 0, 1: the newest steps are 2 at room 2, 4 at room 0, and 5 at room 1.
    actions = torch.tensor([RIGHT, RIGHT, LEFT, LEFT, RIGHT, RIGHT])
    tiles = torch.arange(6, dtype=torch.uint8)[:, None, None].expand(6, 16, 16)
    deduplicated = deduplicate(actions, tiles)
    assert torch.equal(deduplicated.frames[:, 0, 0], torch.tensor([2, 4, 5], dtype=torch.uint8))
    assert torch.equal(deduplicated.actions, torch.tensor([LEFT, RIGHT, RIGHT]))


def test_plan_walk_rejects_an_empty_transcript() -> None:
    with pytest.raises(ValueError, match="no steps"):
        plan_walk(np.zeros(0, dtype=np.uint8))


def test_positions_reject_a_two_dimensional_array() -> None:
    with pytest.raises(ValueError, match="one dimensional"):
        positions_from_actions(np.zeros((2, 3), dtype=np.uint8))


def test_compact_rejects_mismatched_frames_and_actions() -> None:
    with pytest.raises(ValueError, match="do not match"):
        compact(torch.zeros(3, dtype=torch.int64), torch.zeros(2, 16, 16, dtype=torch.uint8))
