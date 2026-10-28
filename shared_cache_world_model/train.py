"""Train one world model on a recording and write results/<run>/summary.json."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
import tyro
from pydantic import BaseModel, DirectoryPath, Field, model_validator

from shared_cache_world_model.data import (
    Recording,
    RecordingIdentity,
    SplitSettings,
    Windows,
    load_recording,
    recording_identity,
    sample_windows,
    validate_matching_worlds,
)
from shared_cache_world_model.model import (
    ModelSettings,
    WorldModel,
    masked_mean,
    next_frame_losses,
    shift_to_targets,
    tile_target_mask,
)
from shared_cache_world_model.probe import score_revisits
from shared_cache_world_model.revisits import validate_gap_bins

# The loss is logged once per this many steps, averaged over the interval.
LOG_EVERY = 50
logger = logging.getLogger(__name__)


class TrainSettings(BaseModel):
    recording: DirectoryPath = Field(description="directory holding steps.npz from the game")
    model: ModelSettings
    experiment: str = Field(default="roomnav_sharedkv_v1", min_length=1)
    seed: int = Field(default=42, description="initialization and batch order")
    # A history of 128 steps is 1,152 tokens per window.
    history: int = Field(default=128, ge=2)
    batch_size: int = Field(default=4, ge=1)
    steps: int = Field(default=2000, ge=1)
    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    checkpoint_every: int = Field(default=500, ge=1)
    # When set, the run ends by scoring revisits in this recording.
    probe_recording: DirectoryPath | None = Field(default=None)
    probe_bins: tuple[int, ...] = Field(default=(1, 8, 16, 32, 64, 128, 256, 512))
    # When set, the probe also runs every this many steps, so a long run shows when retrieval is
    # learned. This value is a multiple of the checkpoint interval.
    probe_every: int | None = Field(default=None, ge=1)
    device: str | None = Field(default=None)
    out: Path = Field(default=Path("results"))
    run_name: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_history_and_probes(self) -> TrainSettings:
        if self.history > self.model.max_history:
            raise ValueError(
                f"history {self.history} exceeds model max_history {self.model.max_history}"
            )
        validate_gap_bins(self.probe_bins)
        if self.probe_every is not None and self.probe_recording is None:
            raise ValueError("probe_every needs a probe_recording")
        if self.probe_every is not None and self.probe_every % self.checkpoint_every != 0:
            raise ValueError(
                f"probe_every {self.probe_every} must be a multiple of checkpoint_every "
                f"{self.checkpoint_every}"
            )
        return self

    def resolved_run_name(self) -> str:
        if self.run_name is not None:
            return self.run_name
        return f"{self.experiment}_{self.model.kind}_h{self.history}_s{self.seed}"


class LossPoint(BaseModel):
    """Mean losses over the steps since the previous point."""

    step: int
    loss: float
    # Revisit targets have appeared earlier in the window. Novel targets have not.
    revisit: float | None
    novel: float | None


class LossTracker:
    """Sums per-cell losses under three masks between log points."""

    def __init__(self, device: torch.device) -> None:
        # Apple GPUs have no float64, so the logged sums use float32 there.
        sum_dtype = torch.float32 if device.type == "mps" else torch.float64
        self.sums = torch.zeros(3, dtype=sum_dtype, device=device)
        self.counts = torch.zeros(3, dtype=torch.int64, device=device)

    def add(self, losses: torch.Tensor, masks: list[torch.Tensor]) -> None:
        # losses: (batch, targets). Each mask is per cell and is shifted onto the targets. The
        # sums stay on the device, so adding costs no synchronization.
        for index, mask in enumerate(masks):
            aligned = shift_to_targets(mask)
            self.sums[index] += losses[aligned].sum()
            self.counts[index] += aligned.sum()

    def flush(self, step: int) -> LossPoint:
        # A mask with no target over the interval gives None, which remains valid JSON.
        means = (self.sums / self.counts.clamp(min=1)).tolist()
        empty = (self.counts == 0).tolist()
        if empty[0]:
            raise ValueError("the aggregate loss tracker has no targets")
        loss = float(means[0])
        revisit = None if empty[1] else float(means[1])
        novel = None if empty[2] else float(means[2])
        self.sums.zero_()
        self.counts.zero_()
        return LossPoint(step=step, loss=loss, revisit=revisit, novel=novel)


class RunSummary(BaseModel):
    run: str
    config: dict[str, object]
    final_loss: float
    loss_log: list[LossPoint]
    # Revisit accuracy by gap bin, when a probe recording was given.
    probe: dict[str, object] | None
    # The same report at each intermediate probe, keyed by the training step.
    checkpoint_probes: dict[int, dict[str, object]]
    recording: RecordingIdentity
    probe_recording: RecordingIdentity | None
    commit: str
    versions: dict[str, str]
    device: str
    started: str
    duration_seconds: float


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("no accelerator found. Pass --device cpu to run on the CPU")
    return accelerator


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def training_batch(
    recording: Recording, settings: TrainSettings, generator: torch.Generator, device: torch.device
) -> Windows:
    """One batch of windows, moved to the device."""
    windows = sample_windows(recording, settings.history, settings.batch_size, generator)
    return Windows(
        frames=windows.frames.to(device),
        actions=windows.actions.to(device),
        revisiting=windows.revisiting.to(device),
    )


def save_checkpoint(
    path: Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    schedule: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    config: dict[str, object],
    recording: RecordingIdentity,
    probe_recording: RecordingIdentity | None,
    step: int,
    loss_log: list[LossPoint],
    checkpoint_probes: dict[int, dict[str, object]],
) -> None:
    # The loss log and the intermediate probes ride along so a resumed run keeps its history.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(),
            "generator": generator.get_state(),
            "config": config,
            "recording": recording.model_dump(mode="json"),
            "probe_recording": (
                None if probe_recording is None else probe_recording.model_dump(mode="json")
            ),
            "step": step,
            "loss_log": [point.model_dump() for point in loss_log],
            "checkpoint_probes": checkpoint_probes,
        },
        path,
    )


def run_probe(
    model: WorldModel, settings: TrainSettings, device: torch.device, run_dir: Path, step: int
) -> dict[str, object]:
    """Score revisits on the probe recording and write probe_<step>.json into the run directory."""
    if settings.probe_recording is None:
        raise ValueError("run_probe needs a probe recording")
    evaluation = load_recording(SplitSettings(recording=settings.probe_recording))
    result = score_revisits(
        model,
        evaluation,
        Path(settings.probe_recording),
        settings.history,
        settings.probe_bins,
        device,
    )
    (run_dir / f"probe_{step}.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result.model_dump(mode="json")["report"]


def main(settings: TrainSettings) -> None:
    started = time.time()
    torch.manual_seed(settings.seed)
    device = resolve_device(settings.device)
    training_identity = recording_identity(Path(settings.recording))
    probe_identity: RecordingIdentity | None = None
    if settings.probe_recording is not None:
        probe_identity = recording_identity(Path(settings.probe_recording))
        validate_matching_worlds(training_identity, probe_identity)
    recording = load_recording(SplitSettings(recording=settings.recording))
    model = WorldModel(settings.model).to(device)
    # AdamW with betas (0.9, 0.95) and decoupled weight decay, the usual transformer recipe.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.lr, betas=(0.9, 0.95), weight_decay=settings.weight_decay
    )
    # Runs of at least three steps warm up for at least two updates, then decay to near zero.
    # One-step and two-step smoke runs keep a constant learning rate.
    if settings.steps < 3:
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        warmup_fraction = max(0.05, 2 / settings.steps)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=settings.lr,
            total_steps=settings.steps,
            pct_start=warmup_fraction,
        )
    run_dir = settings.out / settings.resolved_run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run %s on %s", run_dir.name, device)
    checkpoint_path = run_dir / "checkpoint.pt"
    first_step = 0
    loss_log: list[LossPoint] = []
    checkpoint_probes: dict[int, dict[str, object]] = {}
    config = settings.model_dump(mode="json")
    generator = torch.Generator()
    if checkpoint_path.exists():
        # A machine that stopped mid-run resumes from its last checkpoint.
        state = torch.load(checkpoint_path, map_location=device)
        if state.get("config") != config:
            raise ValueError(f"checkpoint config does not match run {run_dir.name}")
        if state.get("recording") != training_identity.model_dump(mode="json"):
            raise ValueError(f"checkpoint recording does not match run {run_dir.name}")
        expected_probe = None if probe_identity is None else probe_identity.model_dump(mode="json")
        if state.get("probe_recording") != expected_probe:
            raise ValueError(f"checkpoint probe recording does not match run {run_dir.name}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        generator.set_state(state["generator"])
        first_step = int(state["step"])
        if first_step >= settings.steps:
            raise ValueError(f"checkpoint for run {run_dir.name} already reached step {first_step}")
        loss_log = [LossPoint(**point) for point in state["loss_log"]]
        checkpoint_probes = state["checkpoint_probes"]
    else:
        # Window sampling has its own generator so checkpoint resume can restore its exact state.
        generator.manual_seed(settings.seed)
    # bfloat16 halves activation memory on the GPU. The CPU path stays in float32.
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tracker = LossTracker(device)
    model.train()
    loss_value = float("nan")
    for step in range(first_step, settings.steps):
        windows = training_batch(recording, settings, generator, device)
        # logits: (batch, history, 8, 256), computed in bfloat16 on the GPU.
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            logits = model(windows.frames, windows.actions)
        # losses: (batch, (history - 1) * 256) float32, one cross-entropy per cell. The cast
        # to float32 keeps the softmax inside the loss in full precision.
        losses = next_frame_losses(logits.float(), windows.frames)
        # targets: (batch, history * 256) bool. Every cell of every following frame is a target.
        targets = tile_target_mask(torch.ones_like(windows.actions, dtype=torch.bool))
        # The optimizer's loss is the mean over the tile targets.
        loss = masked_mean(losses, targets)
        # set_to_none frees each gradient buffer between steps in place of zeroing it.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Scale the gradients down when their global norm exceeds 1.0, a guard against a rare
        # huge step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Move the learning rate one step along the warmup and decay.
        schedule.step()
        # detach() reads the scalar without holding the autograd graph.
        loss_value = float(loss.detach())
        # The log splits targets by whether their room already appeared inside the window.
        revisit_targets = tile_target_mask(windows.revisiting)
        novel_targets = tile_target_mask(~windows.revisiting)
        tracker.add(losses.detach(), [targets, revisit_targets, novel_targets])
        if (step + 1) % LOG_EVERY == 0:
            point = tracker.flush(step + 1)
            loss_log.append(point)
            revisit_loss = "n/a" if point.revisit is None else f"{point.revisit:.4f}"
            novel_loss = "n/a" if point.novel is None else f"{point.novel:.4f}"
            logger.info(
                "step %d loss %.4f revisit %s novel %s",
                point.step,
                point.loss,
                revisit_loss,
                novel_loss,
            )
        probe_now = settings.probe_every is not None and (step + 1) % settings.probe_every == 0
        if probe_now and settings.probe_recording is not None and step + 1 < settings.steps:
            checkpoint_probes[step + 1] = run_probe(model, settings, device, run_dir, step + 1)
            # The probe switches the model to evaluation mode. Training resumes in training mode.
            model.train()
        if (step + 1) % settings.checkpoint_every == 0 or step + 1 == settings.steps:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                schedule,
                generator,
                config,
                training_identity,
                probe_identity,
                step + 1,
                loss_log,
                checkpoint_probes,
            )
    probe_report: dict[str, object] | None = None
    if settings.probe_recording is not None:
        probe_report = run_probe(model, settings, device, run_dir, settings.steps)
    summary = RunSummary(
        run=run_dir.name,
        config=config,
        final_loss=loss_value,
        loss_log=loss_log,
        probe=probe_report,
        checkpoint_probes=checkpoint_probes,
        recording=training_identity,
        probe_recording=probe_identity,
        commit=current_commit(),
        versions={"torch": torch.__version__, "numpy": version("numpy")},
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        started=datetime.fromtimestamp(started, tz=UTC).isoformat(timespec="seconds"),
        duration_seconds=round(time.time() - started, 1),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2) + "\n"
    )
    logger.info("wrote %s", run_dir / "summary.json")


if __name__ == "__main__":
    # The default logger drops INFO lines. The timestamp shows the step rate in a job log.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(TrainSettings))
