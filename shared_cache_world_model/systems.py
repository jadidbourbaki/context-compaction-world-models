"""Measure each design's cache bytes, decode latency, and the cost and savings of compaction."""

# Every number comes from the trained seed 42 weights of each design, reading windows of the
# evaluation recording. A harness that compacts the transcript
# rebuilds the context before every prediction, so the model prefills the whole context each
# time and reads one prediction at its last step. The saving is one prefill over the compacted
# walk instead of the 128-step window, and the price is one call to the planner.

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
import tyro
from pydantic import BaseModel, DirectoryPath, Field, FilePath

from shared_cache_world_model.agent import load_model
from shared_cache_world_model.compaction import plan_walk
from shared_cache_world_model.data import Recording, SplitSettings, load_recording, model_frames
from shared_cache_world_model.model import Kind, ModelCache, WorldModel
from shared_cache_world_model.probe import ProbeResult
from shared_cache_world_model.train import current_commit

KINDS: tuple[Kind, ...] = ("per_layer", "ced_full", "ced", "shared_cache")


class SystemsSettings(BaseModel):
    recording: DirectoryPath = Field(default=Path("data/roomnav_eval"))
    weights: DirectoryPath = Field(default=Path("weights"))
    compacted_probe: FilePath = Field(
        default=Path("results/roomnav_transition_v1_per_layer_h128_s42/compacted/probe.json"),
        description="a compacted probe whose revisits give the walk lengths and planner inputs",
    )
    contexts: tuple[int, ...] = Field(default=(32, 64, 128))
    batch_sizes: tuple[int, ...] = Field(default=(1, 16))
    repeats: int = Field(default=30, ge=1)
    warmup: int = Field(default=5, ge=0)
    device: str | None = Field(default=None)
    out: Path = Field(default=Path("results/systems"))


class DesignMeasurement(BaseModel):
    kind: str
    parameters: int
    # cache_bytes[context steps]: every tensor the incremental model retains at batch 1.
    cache_bytes: dict[str, int]
    # decode_ms[batch size][context steps]: median latency of one step after that many steps.
    decode_ms: dict[str, dict[str, float]]
    # Median latency of one forward pass over the 128-step window, and the mean over revisits of
    # one forward pass over each compacted walk.
    raw_read_ms: float
    compacted_read_ms: float


class SystemsResult(BaseModel):
    designs: list[DesignMeasurement]
    compacted_steps_mean: float
    planner_ms_mean: float
    planner_ms_max: float
    config: SystemsSettings
    commit: str
    versions: dict[str, str]
    device: str


def synchronize(device: torch.device) -> None:
    # GPU kernels run asynchronously, so a timer stops only after the device finishes its queue.
    if device.type != "cpu":
        torch.accelerator.synchronize()


def median_ms(run: Callable[[], object], device: torch.device, repeats: int, warmup: int) -> float:
    """The median wall time of ``run`` in milliseconds, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        run()
    synchronize(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def recorded_steps(
    recording: Recording, batch: int, steps: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first steps of the first episodes of a recording, one episode per batch row."""
    if steps > recording.steps_per_episode or batch > recording.episodes:
        raise ValueError(f"{batch} episodes of {steps} steps exceed the recording")
    rows = range(batch)
    starts = [row * recording.steps_per_episode for row in rows]
    # frames: (batch, steps, 16, 16) int64 tile ids with the character drawn in.
    frames = torch.stack([model_frames(recording.tiles[start : start + steps]) for start in starts])
    # actions: (batch, steps) int64.
    actions = torch.stack([recording.actions[start : start + steps] for start in starts])
    return frames.long().to(device), actions.to(device)


def load_design(weights: Path, kind: Kind, device: torch.device) -> WorldModel:
    """The trained seed 42 weights of one design, the checkpoint the demo and figures use."""
    checkpoint = weights / f"roomnav_transition_v1_{kind}_h128_s42" / "weights.pt"
    return load_model(checkpoint, device)[0]


def prefill(model: WorldModel, frames: torch.Tensor, actions: torch.Tensor) -> ModelCache | None:
    """Run the incremental path over every step and return the cache it retains."""
    cache = None
    for step in range(actions.shape[1]):
        cache = model.forward_step(frames[:, step], actions[:, step], cache).cache
    return cache


def cache_bytes(cache: ModelCache) -> int:
    """Bytes of every key and value tensor the cache holds."""
    memories = [memory for memory in cache.layers + cache.global_memories if memory is not None]
    return sum(
        tensor.numel() * tensor.element_size()
        for memory in memories
        for tensor in (memory.key, memory.value)
    )


def measure_design(
    kind: Kind,
    settings: SystemsSettings,
    walk_lengths: list[int],
    recording: Recording,
    device: torch.device,
) -> DesignMeasurement:
    model = load_design(settings.weights, kind, device)
    max_history = model.settings.max_history
    measured_bytes: dict[str, int] = {}
    decode_ms: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for batch in settings.batch_sizes:
            decode_ms[str(batch)] = {}
            for context in settings.contexts:
                inputs = recorded_steps(recording, batch, context, device)
                frames = inputs[0]
                actions = inputs[1]
                # The timed step is the last of the context, read against the cache of the
                # steps before it.
                cache = prefill(model, frames[:, :-1], actions[:, :-1])
                last_frame = frames[:, -1]
                last_action = actions[:, -1]
                decode_ms[str(batch)][str(context)] = median_ms(
                    partial(model.forward_step, last_frame, last_action, cache),
                    device,
                    settings.repeats,
                    settings.warmup,
                )
                if batch == 1:
                    full_cache = model.forward_step(last_frame, last_action, cache).cache
                    measured_bytes[str(context)] = cache_bytes(full_cache)
        inputs = recorded_steps(recording, 1, max_history, device)
        frames = inputs[0]
        actions = inputs[1]
        raw_read_ms = median_ms(
            partial(model.forward_last, frames, actions), device, settings.repeats, settings.warmup
        )
        # read_ms[length]: one forward pass over a context of that many steps.
        read_ms: dict[int, float] = {}
        for length in sorted(set(walk_lengths)):
            read_ms[length] = median_ms(
                partial(model.forward_last, frames[:, :length], actions[:, :length]),
                device,
                settings.repeats,
                settings.warmup,
            )
    compacted_read_ms = statistics.mean(read_ms[length] for length in walk_lengths)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return DesignMeasurement(
        kind=kind,
        parameters=parameters,
        cache_bytes=measured_bytes,
        decode_ms=decode_ms,
        raw_read_ms=raw_read_ms,
        compacted_read_ms=compacted_read_ms,
    )


def main(settings: SystemsSettings) -> None:
    device = (
        torch.device(settings.device)
        if settings.device
        else torch.accelerator.current_accelerator()
    )
    if device is None:
        raise RuntimeError("no accelerator found. Pass --device cpu to run on the CPU")
    probe = ProbeResult.model_validate_json(settings.compacted_probe.read_text())
    walk_lengths = [score.context_steps for score in probe.scores]
    recording = load_recording(SplitSettings(recording=settings.recording))
    # The planner runs on the CPU over the episode prefix before each probed revisit.
    planner_ms = []
    for score in probe.scores:
        episode_start = score.step - recording.offset_in_episode(score.step)
        prefix = recording.actions[episode_start : score.step].numpy()
        start = time.perf_counter()
        plan_walk(prefix)
        planner_ms.append((time.perf_counter() - start) * 1000)
    designs = [measure_design(kind, settings, walk_lengths, recording, device) for kind in KINDS]
    result = SystemsResult(
        designs=designs,
        compacted_steps_mean=statistics.mean(walk_lengths),
        planner_ms_mean=statistics.mean(planner_ms),
        planner_ms_max=max(planner_ms),
        config=settings,
        commit=current_commit(),
        versions={"torch": torch.__version__, "numpy": version("numpy")},
        device=str(device),
    )
    settings.out.mkdir(parents=True, exist_ok=True)
    (settings.out / "summary.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(json.dumps(result.model_dump(mode="json", exclude={"config"}), indent=2))


if __name__ == "__main__":
    main(tyro.cli(SystemsSettings))
