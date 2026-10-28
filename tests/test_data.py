from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, assume, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from shared_cache_world_model.data import (
    SplitSettings,
    episode_length,
    load_recording,
    model_frames,
    recording_identity,
    sample_windows,
    validate_matching_worlds,
    window_revisits,
)
from shared_cache_world_model.world import CHARACTER, ROOM_COLS, ROOM_ROWS, WorldSettings, main


def test_episode_length_rejects_uneven_episodes() -> None:
    assert episode_length(np.array([0, 0, 1, 1])) == 2
    with pytest.raises(ValueError, match="episodes run from"):
        episode_length(np.array([0, 0, 0, 1]))


def test_a_recording_without_episodes_is_rejected(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "steps.npz",
        tiles=np.zeros((4, ROOM_ROWS, ROOM_COLS), dtype=np.uint8),
        actions=np.zeros(4, dtype=np.uint8),
        room=np.zeros(4, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="has no episode array"):
        load_recording(SplitSettings(recording=tmp_path))


@given(
    world_seed=st.integers(min_value=0, max_value=2**32 - 1),
    sample_seed=st.integers(min_value=0, max_value=2**32 - 1),
    history=st.integers(min_value=2, max_value=10),
    batch_size=st.integers(min_value=1, max_value=8),
)
@hypothesis_settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_windows_stay_inside_one_episode(
    tmp_path: Path,
    world_seed: int,
    sample_seed: int,
    history: int,
    batch_size: int,
) -> None:
    main(
        WorldSettings(
            out=tmp_path,
            episodes=5,
            steps_per_episode=20,
            seed=world_seed,
            video_steps=0,
        )
    )
    recording = load_recording(SplitSettings(recording=tmp_path))
    generator = torch.Generator().manual_seed(sample_seed)
    windows = sample_windows(
        recording,
        history=history,
        batch_size=batch_size,
        generator=generator,
    )
    assert windows.frames.shape == (batch_size, history, ROOM_ROWS, ROOM_COLS)
    assert windows.actions.shape == (batch_size, history)
    assert windows.revisiting.shape == (batch_size, history)
    for row in range(batch_size):
        matching_starts = [
            start
            for start in range(recording.steps - history + 1)
            if torch.equal(
                windows.frames[row],
                model_frames(recording.tiles[start : start + history]).long(),
            )
            and torch.equal(
                windows.actions[row],
                recording.actions[start : start + history],
            )
        ]
        assert matching_starts
        # Every matching slice stays inside one episode.
        assert all(start // 20 == (start + history - 1) // 20 for start in matching_starts)


def test_window_revisits_uses_only_earlier_rooms_in_the_same_window() -> None:
    rooms = torch.tensor([[1, 2, 1, 3, 2], [4, 4, 5, 4, 5]])
    revisiting = window_revisits(rooms)
    assert revisiting.tolist() == [
        [False, False, True, False, True],
        [False, True, False, True, True],
    ]


def test_a_history_longer_than_an_episode_is_rejected(tmp_path: Path) -> None:
    main(WorldSettings(out=tmp_path, episodes=3, steps_per_episode=8, video_steps=0))
    recording = load_recording(SplitSettings(recording=tmp_path))
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="shorter than history"):
        sample_windows(recording, history=9, batch_size=2, generator=generator)


@given(
    training_seed=st.integers(min_value=0, max_value=2**32 - 1),
    evaluation_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@hypothesis_settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_recording_identity_hashes_and_matches_generated_worlds(
    tmp_path: Path, training_seed: int, evaluation_seed: int
) -> None:
    assume(training_seed != evaluation_seed)
    training_dir = tmp_path / "train"
    evaluation_dir = tmp_path / "eval"
    main(
        WorldSettings(
            out=training_dir,
            episodes=2,
            steps_per_episode=20,
            seed=training_seed,
            video_steps=0,
        )
    )
    main(
        WorldSettings(
            out=evaluation_dir,
            episodes=2,
            steps_per_episode=20,
            seed=evaluation_seed,
            video_steps=0,
        )
    )
    training = recording_identity(training_dir)
    evaluation = recording_identity(evaluation_dir)
    assert len(training.sha256) == 64
    assert training.sha256 != evaluation.sha256
    validate_matching_worlds(training, evaluation)
    same_seed = evaluation.model_copy(
        update={"settings": evaluation.settings.model_copy(update={"seed": training_seed})}
    )
    with pytest.raises(ValueError, match="share seed"):
        validate_matching_worlds(training, same_seed)


def test_recording_identity_rejects_an_array_with_the_wrong_shape(tmp_path: Path) -> None:
    main(WorldSettings(out=tmp_path, episodes=2, steps_per_episode=20, video_steps=0))
    archive_path = tmp_path / "steps.npz"
    with np.load(archive_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["tiles"] = arrays["tiles"][:-1]
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="tiles has shape"):
        recording_identity(tmp_path)


def test_recording_identity_rejects_an_unexpected_array(tmp_path: Path) -> None:
    main(WorldSettings(out=tmp_path, episodes=2, steps_per_episode=20, video_steps=0))
    archive_path = tmp_path / "steps.npz"
    with np.load(archive_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["screen"] = arrays["tiles"].copy()
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="unexpected arrays: screen"):
        recording_identity(tmp_path)


def test_recording_identity_rejects_an_unversioned_recording(tmp_path: Path) -> None:
    main(WorldSettings(out=tmp_path, episodes=2, steps_per_episode=20, video_steps=0))
    meta_path = tmp_path / "meta.json"
    metadata = json.loads(meta_path.read_text())
    del metadata["format_version"]
    meta_path.write_text(json.dumps(metadata) + "\n")
    with pytest.raises(ValidationError, match="format_version"):
        recording_identity(tmp_path)


def test_recording_identity_rejects_a_transition_that_disagrees_with_its_action(
    tmp_path: Path,
) -> None:
    main(WorldSettings(out=tmp_path, episodes=2, steps_per_episode=20, video_steps=0))
    archive_path = tmp_path / "steps.npz"
    with np.load(archive_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["room"][1] = arrays["room"][0]
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="transitions do not match"):
        recording_identity(tmp_path)


def test_recording_identity_rejects_a_character_in_room_tiles(tmp_path: Path) -> None:
    main(WorldSettings(out=tmp_path, episodes=2, steps_per_episode=20, video_steps=0))
    archive_path = tmp_path / "steps.npz"
    with np.load(archive_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["tiles"][0, 0, 0] = CHARACTER
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="contain the character"):
        recording_identity(tmp_path)
