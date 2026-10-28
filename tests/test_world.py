from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from shared_cache_world_model.baselines import measure
from shared_cache_world_model.data import SplitSettings, load_recording
from shared_cache_world_model.revisits import find_revisits
from shared_cache_world_model.world import (
    CHARACTER,
    DOOR,
    ROCK,
    ROOM_COLS,
    ROOM_ROWS,
    TREE,
    WALL,
    WATER,
    Doors,
    WorldSettings,
    add_character,
    generate_room,
    generate_world,
    main,
    move_room,
    record,
    room_doors,
    room_keys,
    valid_room_actions,
)


def small(**overrides: object) -> WorldSettings:
    defaults: dict[str, object] = {
        "out": Path("unused"),
        "episodes": 3,
        "steps_per_episode": 40,
        "rooms_across": 3,
        "rooms_down": 3,
        "video_steps": 0,
    }
    defaults.update(overrides)
    return WorldSettings(**defaults)


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    north=st.integers(min_value=2, max_value=ROOM_COLS - 3),
    east=st.integers(min_value=2, max_value=ROOM_ROWS - 3),
)
@hypothesis_settings(max_examples=20, deadline=None)
def test_a_room_is_walled_with_doors_on_the_sides_that_have_neighbours(
    seed: int, north: int, east: int
) -> None:
    rng = np.random.default_rng(seed)
    doors = Doors(north=north, south=None, west=None, east=east)
    tiles = generate_room(rng, small(), doors)
    assert tiles.shape == (ROOM_ROWS, ROOM_COLS)
    assert int(tiles[0, north]) == DOOR
    assert int(tiles[east, ROOM_COLS - 1]) == DOOR
    # The sides without a neighbour stay solid wall.
    assert set(tiles[ROOM_ROWS - 1, :].tolist()) == {WALL}
    assert set(tiles[:, 0].tolist()) == {WALL}


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    patches=st.integers(min_value=0, max_value=50),
    radius=st.integers(min_value=1, max_value=4),
)
@hypothesis_settings(max_examples=30, deadline=None)
def test_every_door_of_a_generated_room_reaches_the_centre(
    seed: int, patches: int, radius: int
) -> None:
    world_settings = small(patches_per_room=patches, patch_radius=radius)
    rng = np.random.default_rng(seed)
    doors = Doors(north=4, south=9, west=6, east=11)
    tiles = generate_room(rng, world_settings, doors)
    graph = nx.grid_2d_graph(ROOM_ROWS, ROOM_COLS)
    blocking = {WALL, ROCK, TREE, WATER}
    blocked = [cell for cell in graph if int(tiles[cell]) in blocking]
    graph.remove_nodes_from(blocked)
    centre = (ROOM_ROWS // 2, ROOM_COLS // 2)
    door_entries = [(1, 4), (ROOM_ROWS - 2, 9), (6, 1), (11, ROOM_COLS - 2)]
    assert all(nx.has_path(graph, entry, centre) for entry in door_entries)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@hypothesis_settings(max_examples=20, deadline=None)
def test_neighbouring_rooms_share_a_door_position(seed: int) -> None:
    settings = small()
    rng = np.random.default_rng(seed)
    world = generate_world(rng, settings)
    assert world.tiles.shape == (3, 3, ROOM_ROWS, ROOM_COLS)
    left = room_doors(settings, world, 1, 0)
    right = room_doors(settings, world, 1, 1)
    assert left.east == right.west
    assert left.west is None
    upper = room_doors(settings, world, 0, 1)
    lower = room_doors(settings, world, 1, 1)
    assert upper.south == lower.north


def test_only_grid_neighbours_are_valid_room_actions() -> None:
    settings = small()
    assert valid_room_actions(settings, room=0) == [1, 3]
    assert move_room(settings, room=0, action=1) == 3
    assert move_room(settings, room=0, action=3) == 1
    with pytest.raises(ValueError, match="invalid from room 0"):
        move_room(settings, room=0, action=0)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@hypothesis_settings(max_examples=10, deadline=None)
def test_a_recording_covers_rooms_and_revisits_them(seed: int) -> None:
    settings = small(episodes=4, steps_per_episode=120, seed=seed)
    recording = record(settings)
    steps = settings.episodes * settings.steps_per_episode
    assert recording.tiles.shape == (steps, ROOM_ROWS, ROOM_COLS)
    assert recording.episode.tolist() == sorted(recording.episode.tolist())
    # The stored room omits the character. A model window adds it at the fixed centre cell.
    assert int((recording.tiles == CHARACTER).sum()) == 0
    screen = add_character(recording.tiles)
    assert int((screen == CHARACTER).sum()) == steps
    assert bool((screen[:, ROOM_ROWS // 2, ROOM_COLS // 2] == CHARACTER).all())
    # The route leaves its starting room, and it comes back to rooms it has already seen.
    assert len(set(recording.room.tolist())) > 1
    for index in range(steps - 1):
        if recording.episode[index] == recording.episode[index + 1]:
            expected = move_room(
                settings,
                int(recording.room[index]),
                int(recording.actions[index]),
            )
            assert int(recording.room[index + 1]) == expected
    # A grid with at least two rows and columns always offers an exit besides the arrival edge.
    for index in range(2, steps):
        same_episode = recording.episode[index] == recording.episode[index - 2]
        if same_episode:
            assert recording.room[index] != recording.room[index - 2]
    revisits = find_revisits(recording.episode, recording.room)
    assert len(revisits) > 0
    # A revisit never joins two episodes, because a room key carries its episode.
    for revisit in revisits:
        assert recording.episode[revisit.step] == recording.episode[revisit.previous_step]


def test_a_written_recording_loads(tmp_path: Path) -> None:
    settings = small(episodes=3, steps_per_episode=30, out=tmp_path)
    main(settings)
    loaded = load_recording(SplitSettings(recording=tmp_path))
    assert loaded.steps == 90
    assert loaded.steps_per_episode == 30
    assert loaded.episodes == 3
    assert not (tmp_path / "steps.gif").exists()


def test_a_requested_preview_is_written(tmp_path: Path) -> None:
    settings = small(episodes=1, steps_per_episode=2, out=tmp_path, video_steps=2)
    main(settings)
    assert (tmp_path / "steps.gif").exists()


def test_baselines_score_every_transition_and_revisit(tmp_path: Path) -> None:
    settings = small(episodes=2, steps_per_episode=80, out=tmp_path)
    main(settings)
    archive = np.load(tmp_path / "steps.npz")
    revisits = find_revisits(archive["episode"], archive["room"])
    baselines = measure(load_recording(SplitSettings(recording=tmp_path)))
    assert baselines["transition_frames"] == 2 * 79
    assert baselines["revisits"] == len(revisits)


def test_a_world_with_too_many_rooms_is_rejected() -> None:
    with pytest.raises(ValidationError, match="rooms exceed"):
        WorldSettings(out=Path("unused"), rooms_across=17, rooms_down=17)


@given(
    first_seed=st.integers(min_value=0, max_value=2**32 - 1),
    second_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@hypothesis_settings(max_examples=10, deadline=None)
def test_room_keys_of_two_seeds_never_meet(first_seed: int, second_seed: int) -> None:
    assume(first_seed != second_seed)
    first = record(small(seed=first_seed))
    second = record(small(seed=second_seed))
    assert not set(room_keys(first.episode, first.room).tolist()) & set(
        room_keys(second.episode, second.room).tolist()
    )
