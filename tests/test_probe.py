from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from shared_cache_world_model.data import SplitSettings, load_recording, model_frames
from shared_cache_world_model.model import ModelSettings
from shared_cache_world_model.probe import frame_accuracy
from shared_cache_world_model.revisits import find_revisits
from shared_cache_world_model.world import INTERIOR, TILE_VOCAB, WorldSettings, main


class Parrot:
    """A stand-in model whose output at every position is the token it was given there."""

    def __init__(self) -> None:
        # The probe reads the longest context the model accepts. An episode here is 96 steps.
        self.settings = ModelSettings(kind="per_layer", d_model=32, n_heads=4, max_history=96)

    def eval(self) -> None:
        return None

    def __call__(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # logits: (1, steps, 8, 256), one at each input cell's own tile type.
        logits = torch.nn.functional.one_hot(frames.long(), TILE_VOCAB)
        logits = logits.reshape(frames.shape[0], frames.shape[1], 256, TILE_VOCAB)
        return logits.permute(0, 1, 3, 2).float()


def a_recording(path: Path, seed: int) -> None:
    main(WorldSettings(out=path, episodes=2, steps_per_episode=96, seed=seed, video_steps=0))


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@hypothesis_settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_the_probe_reads_the_step_before_the_entering_frame(tmp_path: Path, seed: int) -> None:
    a_recording(tmp_path, seed)
    recording = load_recording(SplitSettings(recording=tmp_path))
    archive = np.load(tmp_path / "steps.npz")
    history = 16
    revisits = [
        revisit
        for revisit in find_revisits(archive["episode"], archive["room"])
        if recording.offset_in_episode(revisit.step) >= history - 1
    ]
    assert revisits
    interior = torch.from_numpy(INTERIOR.reshape(-1))
    for revisit in revisits[:5]:
        result = frame_accuracy(Parrot(), recording, revisit, history, torch.device("cpu"))  # type: ignore[arg-type]
        scored = result.accuracy
        assert result.context_steps == history
        # A model that repeats its input scores the previous frame against the entering one,
        # because the probe reads the step before the entering step. Reading the entering step
        # itself would score 1.0 on every revisit.
        entering = model_frames(recording.tiles[revisit.step]).reshape(-1).long()
        room = recording.tiles[revisit.step].reshape(-1).long()
        previous = model_frames(recording.tiles[revisit.step - 1]).reshape(-1).long()
        cells = (entering == room) & interior
        expected = float((previous[cells] == entering[cells]).float().mean())
        assert scored == pytest.approx(expected)
        assert scored < 1.0
        shuffled = frame_accuracy(
            Parrot(),  # type: ignore[arg-type]
            recording,
            revisit,
            history,
            torch.device("cpu"),
            shuffle_distant_history=True,
        )
        assert shuffled.accuracy == pytest.approx(scored)
        # Every reduced context ends with the current room's newest frame and then the entering
        # step, so the echo model reads the same frame it read in the raw window. If a context
        # ended anywhere else, or the entering step were misplaced, this would move.
        episode_start = revisit.step - recording.offset_in_episode(revisit.step)
        distinct_rooms = len(set(recording.rooms[episode_start : revisit.step].tolist()))
        context_steps = {}
        for context_mode in ("compacted", "truncated", "deduplicated"):
            for shuffle in (False, True):
                reduced = frame_accuracy(
                    Parrot(),  # type: ignore[arg-type]
                    recording,
                    revisit,
                    history,
                    torch.device("cpu"),
                    shuffle_distant_history=shuffle,
                    context_mode=context_mode,
                )
                assert reduced.accuracy == pytest.approx(scored)
            context_steps[context_mode] = reduced.context_steps
        # The walk covers each distinct room at most twice, then the entering step follows. The
        # truncated window matches the walk's length up to the episode's start, and deduplication
        # keeps each room once.
        assert 2 <= context_steps["compacted"] <= 2 * distinct_rooms + 1
        prefix_steps = revisit.step - episode_start
        expected_truncated = min(context_steps["compacted"], prefix_steps + 1)
        assert context_steps["truncated"] == expected_truncated
        assert context_steps["deduplicated"] == distinct_rooms + 1


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@hypothesis_settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_compacting_when_full_stays_inside_the_window(tmp_path: Path, seed: int) -> None:
    # 16 rooms keep every walk well inside a 64-step window.
    main(
        WorldSettings(
            out=tmp_path,
            episodes=2,
            steps_per_episode=96,
            rooms_across=4,
            rooms_down=4,
            seed=seed,
            video_steps=0,
        )
    )
    recording = load_recording(SplitSettings(recording=tmp_path))
    archive = np.load(tmp_path / "steps.npz")
    history = 64
    revisits = [
        revisit
        for revisit in find_revisits(archive["episode"], archive["room"])
        if recording.offset_in_episode(revisit.step) >= history - 1
    ]
    assert revisits
    for revisit in revisits[-5:]:
        raw = frame_accuracy(
            Parrot(),  # type: ignore[arg-type]
            recording,
            revisit,
            history,
            torch.device("cpu"),
        )
        rolling = frame_accuracy(
            Parrot(),  # type: ignore[arg-type]
            recording,
            revisit,
            history,
            torch.device("cpu"),
            context_mode="compacted_when_full",
        )
        # The echo model scores the same only if the context ends where the raw window does.
        assert rolling.accuracy == pytest.approx(raw.accuracy)
        assert rolling.context_steps <= history
