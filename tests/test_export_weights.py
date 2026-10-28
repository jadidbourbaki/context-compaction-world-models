from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shared_cache_world_model.export_weights import KEPT_ENTRIES, WEIGHTS_NAME, export_run
from shared_cache_world_model.model import ModelSettings, WorldModel


def test_exported_weights_load_into_the_model_without_the_optimizer(tmp_path: Path) -> None:
    settings = ModelSettings(kind="ced_full", d_model=32, n_heads=4, n_layers=4, max_history=6)
    model = WorldModel(settings)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"model": settings.model_dump(mode="json"), "history": 6},
            "recording": {"sha256": "train"},
            "probe_recording": {"sha256": "eval"},
            "step": 10,
        },
        checkpoint,
    )
    export_run(checkpoint, tmp_path / "run")
    exported = torch.load(tmp_path / "run" / WEIGHTS_NAME, weights_only=False)
    assert set(exported) == set(KEPT_ENTRIES)
    # The probe rebuilds the model from the stored settings and loads the weights strictly.
    rebuilt = WorldModel(ModelSettings.model_validate(exported["config"]["model"]))
    rebuilt.load_state_dict(exported["model"])
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[name], tensor)


def test_export_rejects_a_checkpoint_without_its_recording_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {}, "config": {}, "step": 1}, checkpoint)
    with pytest.raises(ValueError, match="lacks recording, probe_recording"):
        export_run(checkpoint, tmp_path / "run")
