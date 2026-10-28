from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from shared_cache_world_model.data import TILES_PER_FRAME
from shared_cache_world_model.model import ModelSettings, tile_target_mask
from shared_cache_world_model.train import LossTracker, TrainSettings
from shared_cache_world_model.train import main as train
from shared_cache_world_model.world import WorldSettings
from shared_cache_world_model.world import main as write_world


def test_loss_tracker_uses_null_for_an_empty_target_group() -> None:
    tracker = LossTracker(torch.device("cpu"))
    losses = torch.ones(1, TILES_PER_FRAME)
    all_steps = torch.tensor([[True, True]])
    no_steps = torch.tensor([[False, False]])
    all_targets = tile_target_mask(all_steps)
    no_targets = tile_target_mask(no_steps)
    tracker.add(losses, [all_targets, no_targets, all_targets])
    point = tracker.flush(step=1)
    assert point.loss == pytest.approx(1.0)
    assert point.revisit is None
    assert point.novel == pytest.approx(1.0)


def test_checkpoint_rejects_changed_training_data(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    write_world(
        WorldSettings(
            out=recording_dir,
            episodes=2,
            steps_per_episode=8,
            video_steps=0,
        )
    )
    settings = TrainSettings(
        recording=recording_dir,
        model=ModelSettings(
            kind="per_layer",
            d_model=16,
            n_heads=2,
            n_layers=2,
            max_history=4,
        ),
        history=4,
        batch_size=1,
        steps=1,
        checkpoint_every=1,
        device="cpu",
        out=tmp_path / "results",
        run_name="checksum",
    )
    train(settings)
    archive_path = recording_dir / "steps.npz"
    with np.load(archive_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    original_tile = int(arrays["tiles"][0, 1, 1])
    arrays["tiles"][0, 1, 1] = (original_tile + 1) % 7
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="checkpoint recording does not match"):
        train(settings)
