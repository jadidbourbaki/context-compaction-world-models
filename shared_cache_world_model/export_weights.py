"""Strip training checkpoints down to the weights and identity a probe needs."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import tyro
from pydantic import BaseModel, DirectoryPath, Field

# A probe reads these entries and nothing else. The optimizer, scheduler, and sampler state
# that resuming a run needs make up about two thirds of a checkpoint and are left out.
KEPT_ENTRIES = ("model", "config", "recording", "probe_recording", "step")
WEIGHTS_NAME = "weights.pt"
logger = logging.getLogger(__name__)


class ExportSettings(BaseModel):
    results: DirectoryPath = Field(default=Path("results"))
    out: Path = Field(description="directory that receives one folder per run")
    run_names: tuple[str, ...] = Field(min_length=1, description="result directories to export")


def export_run(checkpoint: Path, destination: Path) -> None:
    """Write one run's weights and identity to ``destination`` / weights.pt."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing = [entry for entry in KEPT_ENTRIES if entry not in state]
    if missing:
        raise ValueError(f"{checkpoint} lacks {', '.join(missing)}")
    kept = {entry: state[entry] for entry in KEPT_ENTRIES}
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(kept, destination / WEIGHTS_NAME)


def main(settings: ExportSettings) -> None:
    for run_name in settings.run_names:
        checkpoint = Path(settings.results) / run_name / "checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"run {run_name} has no checkpoint at {checkpoint}")
        destination = settings.out / run_name
        export_run(checkpoint, destination)
        summary = Path(settings.results) / run_name / "summary.json"
        (destination / "summary.json").write_bytes(summary.read_bytes())
        logger.info("exported %s", run_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(ExportSettings))
