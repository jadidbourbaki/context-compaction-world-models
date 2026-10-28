"""Measure the inference latency of every step of one session under each agent harness policy."""

# A session is one evaluation episode of 512 steps. At every step the agent harness builds the
# context it sends, which for a compacting policy means planning a walk on the CPU, and the
# world model either decodes one step from the KV cache or prefills its whole context. We time
# the harness work at every step, time the model at every context length it meets, and add up
# the cost of each step from the operations that step performs.

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Literal, cast

import torch
import tyro
from pydantic import BaseModel, DirectoryPath, Field

from shared_cache_world_model.compaction import Transcript, compact
from shared_cache_world_model.data import Recording, SplitSettings, load_recording, model_frames
from shared_cache_world_model.model import Kind, ModelSettings
from shared_cache_world_model.systems import KINDS, load_design, median_ms
from shared_cache_world_model.train import current_commit

Policy = Literal["raw", "compacted", "compacted_when_full"]
POLICIES: tuple[Policy, ...] = ("raw", "compacted", "compacted_when_full")


class SessionSettings(BaseModel):
    recording: DirectoryPath = Field(default=Path("data/roomnav_eval"))
    weights: DirectoryPath = Field(default=Path("weights"))
    episode: int = Field(default=0, ge=0)
    repeats: int = Field(default=10, ge=1)
    warmup: int = Field(default=3, ge=0)
    device: str | None = Field(default=None)
    out: Path = Field(default=Path("results/session"))


class SessionPlan(BaseModel):
    # context_steps[policy][step]: the steps the model reads at that step of the session.
    context_steps: dict[str, list[int]]
    # prefills[policy][step]: whether that step prefills its context or decodes one step.
    prefills: dict[str, list[bool]]
    # harness_ms[policy][step]: the CPU time the agent harness takes to build the context it sends
    # at that step, which includes planning a walk when the policy compacts.
    harness_ms: dict[str, list[float]]


class DesignSession(BaseModel):
    kind: str
    # step_ms[policy][step]: the inference latency of that step of the session.
    step_ms: dict[str, list[float]]


class SessionResult(BaseModel):
    plan: SessionPlan
    designs: list[DesignSession]
    config: SessionSettings
    commit: str
    versions: dict[str, str]
    device: str


def median_harness_ms(build: Callable[[], object]) -> tuple[object, float]:
    """What one harness step builds and the median of three timings of building it."""
    timings = []
    built: object = None
    for _ in range(3):
        start = time.perf_counter()
        built = build()
        timings.append((time.perf_counter() - start) * 1000)
    return built, statistics.median(timings)


def plan_session(actions: torch.Tensor, tiles: torch.Tensor, history: int) -> SessionPlan:
    """What each policy does at every step of one episode, with the harness time of each step."""
    steps = int(actions.shape[0])
    # frames: one (16, 16) uint8 room per step, the transcript as the agent harness stores it.
    frames = list(tiles)
    context_steps: dict[str, list[int]] = {policy: [] for policy in POLICIES}
    prefills: dict[str, list[bool]] = {policy: [] for policy in POLICIES}
    harness_ms: dict[str, list[float]] = {policy: [] for policy in POLICIES}
    # The raw context window decodes from its KV cache until it holds history steps, so the
    # harness only sends the newest step. The step positions are absolute, so after that the
    # harness stacks the latest history steps on every step and the model prefills them again.
    for step in range(steps):
        low = max(0, step + 1 - history)
        if step >= history:
            built = median_harness_ms(partial(torch.stack, frames[low : step + 1]))
        else:
            built = median_harness_ms(partial(torch.stack, frames[step : step + 1]))
        context_steps["raw"].append(step + 1 - low)
        prefills["raw"].append(step >= history)
        harness_ms["raw"].append(built[1])
    # Compact every step plans a walk over the whole transcript, gathers its frames, and has the
    # model prefill it at every step.
    for step in range(steps):
        built = median_harness_ms(partial(compact, actions[: step + 1], tiles[: step + 1]))
        context = cast(Transcript, built[0])
        context_steps["compacted"].append(int(context.frames.shape[0]))
        prefills["compacted"].append(True)
        harness_ms["compacted"].append(built[1])
    # Compact when full follows compaction.compact_when_full: raw steps until the context holds
    # history steps, then one walk over the transcript so far, then raw steps again.
    held = 0
    for step in range(steps):
        if held == history:
            built = median_harness_ms(partial(compact, actions[:step], tiles[:step]))
            context = cast(Transcript, built[0])
            held = int(context.frames.shape[0]) + 1
            prefills["compacted_when_full"].append(True)
        else:
            built = median_harness_ms(partial(torch.stack, frames[step : step + 1]))
            held += 1
            prefills["compacted_when_full"].append(False)
        harness_ms["compacted_when_full"].append(built[1])
        context_steps["compacted_when_full"].append(held)
    return SessionPlan(context_steps=context_steps, prefills=prefills, harness_ms=harness_ms)


def measure_design(
    kind: Kind,
    plan: SessionPlan,
    settings: SessionSettings,
    recording: Recording,
    device: torch.device,
) -> DesignSession:
    model = load_design(settings.weights, kind, device)
    longest = max(max(lengths) for lengths in plan.context_steps.values())
    # The timed contexts read the session's own frames. frames: (1, longest, 16, 16) int64.
    start = settings.episode * recording.steps_per_episode
    frames = model_frames(recording.tiles[start : start + longest])[None].long().to(device)
    # actions: (1, longest) int64.
    actions = recording.actions[start : start + longest][None].to(device)
    with torch.inference_mode():
        # prefill_ms[length]: one forward pass over a context of that many steps.
        prefill_ms: dict[int, float] = {}
        for policy in POLICIES:
            for index in range(len(plan.prefills[policy])):
                length = plan.context_steps[policy][index]
                if plan.prefills[policy][index] and length not in prefill_ms:
                    prefill_ms[length] = median_ms(
                        partial(model.forward_last, frames[:, :length], actions[:, :length]),
                        device,
                        settings.repeats,
                        settings.warmup,
                    )
        # decode_ms[length]: one step read against a KV cache of length - 1 steps.
        decode_ms: dict[int, float] = {}
        cache = None
        for length in range(1, model.settings.max_history + 1):
            frame = frames[:, length - 1]
            action = actions[:, length - 1]
            decode_ms[length] = median_ms(
                partial(model.forward_step, frame, action, cache),
                device,
                settings.repeats,
                settings.warmup,
            )
            cache = model.forward_step(frame, action, cache).cache
    step_ms: dict[str, list[float]] = {}
    for policy in POLICIES:
        step_ms[policy] = []
        for index in range(len(plan.prefills[policy])):
            length = plan.context_steps[policy][index]
            model_ms = prefill_ms[length] if plan.prefills[policy][index] else decode_ms[length]
            step_ms[policy].append(plan.harness_ms[policy][index] + model_ms)
    return DesignSession(kind=kind, step_ms=step_ms)


def main(settings: SessionSettings) -> None:
    device = (
        torch.device(settings.device)
        if settings.device
        else torch.accelerator.current_accelerator()
    )
    if device is None:
        raise RuntimeError("no accelerator found. Pass --device cpu to run on the CPU")
    recording = load_recording(SplitSettings(recording=settings.recording))
    if settings.episode >= recording.episodes:
        raise ValueError(f"episode {settings.episode} of {recording.episodes} does not exist")
    start = settings.episode * recording.steps_per_episode
    actions = recording.actions[start : start + recording.steps_per_episode]
    history = ModelSettings(kind="per_layer").max_history
    tiles = recording.tiles[start : start + recording.steps_per_episode]
    plan = plan_session(actions, tiles, history)
    designs = [measure_design(kind, plan, settings, recording, device) for kind in KINDS]
    result = SessionResult(
        plan=plan,
        designs=designs,
        config=settings,
        commit=current_commit(),
        versions={"torch": torch.__version__, "numpy": version("numpy")},
        device=str(device),
    )
    settings.out.mkdir(parents=True, exist_ok=True)
    (settings.out / "summary.json").write_text(result.model_dump_json(indent=2) + "\n")
    for design in designs:
        means = {
            policy: round(statistics.mean(design.step_ms[policy][history:]), 2)
            for policy in POLICIES
        }
        print(design.kind, json.dumps(means))


if __name__ == "__main__":
    main(tyro.cli(SessionSettings))
