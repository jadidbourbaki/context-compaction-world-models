"""Score a trained model on revisits: room tiles right when the character re-enters a room."""

# Three reference points frame the scores. Guessing each position's most common tile gets about
# 0.41. Copying the previous step's frame gets about 0.31, since that frame shows the room the
# character is leaving. Copying the frame from the last visit to the room gets 1.00, because a
# generated room never changes once its episode has made it.

from __future__ import annotations

import json
import random
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator

from shared_cache_world_model.compaction import (
    Transcript,
    compact,
    compact_when_full,
    deduplicate,
    plan_walk,
)
from shared_cache_world_model.data import (
    TILES_PER_FRAME,
    Recording,
    SplitSettings,
    load_recording,
    model_frames,
    recording_identity,
    validate_matching_worlds,
)
from shared_cache_world_model.model import (
    LOCAL_STEPS,
    ModelSettings,
    WorldModel,
)
from shared_cache_world_model.revisits import Revisit, find_revisits, validate_gap_bins
from shared_cache_world_model.world import INTERIOR

# raw: the last ``history`` steps. compacted: the planned walk over every room of the episode so
# far. truncated: the last steps up to the compacted walk's length, a recency baseline at the same
# budget. deduplicated: each room's newest step in visit order, the walk's rooms without the walk.
# compacted_when_full: raw steps until the window fills, then a walk, then raw steps again.
ContextMode = Literal["raw", "compacted", "truncated", "deduplicated", "compacted_when_full"]


class ProbeSettings(BaseModel):
    recording: DirectoryPath = Field(description="an evaluation recording, from a different seed")
    training_recording: DirectoryPath = Field(description="the recording the model trained on")
    checkpoint: FilePath = Field(description="checkpoint.pt written by train")
    model: ModelSettings
    history: int = Field(default=128, ge=2)
    bins: tuple[int, ...] = Field(default=(1, 8, 16, 32, 64, 128, 256, 512))
    limit: int | None = Field(default=None, ge=1, description="revisits to score, for smoke tests")
    per_bin: int | None = Field(default=40, ge=1, description="revisits scored per gap bin")
    context_mode: ContextMode = Field(default="raw")
    device: str | None = Field(default=None)
    out: Path = Field(description="directory for probe.json")

    @model_validator(mode="after")
    def validate_history_and_bins(self) -> ProbeSettings:
        if self.history > self.model.max_history:
            raise ValueError(
                f"history {self.history} exceeds model max_history {self.model.max_history}"
            )
        validate_gap_bins(self.bins)
        return self


class RevisitScore(BaseModel):
    step: int
    gap: int
    accuracy: float
    shuffled_accuracy: float
    retrieval_gain: float
    history_available: bool
    # How many steps the model read, the fixed history for a raw window and the kept steps plus
    # the entering step for the other contexts.
    context_steps: int


class BinSummary(BaseModel):
    mean: float
    shuffled_mean: float
    gain_mean: float
    history_available_fraction: float
    n: int


class ProbeResult(BaseModel):
    # report[gap bin] holds the mean accuracy and the count of revisits in that bin.
    report: dict[str, BinSummary]
    scores: list[RevisitScore]


class FrameScore(BaseModel):
    """One revisit's accuracy and the number of steps the model read to produce it."""

    accuracy: float
    context_steps: int


def probe_context(
    recording: Recording, revisit: Revisit, history: int, context_mode: ContextMode
) -> Transcript:
    """The frames and actions the model reads to predict the entering room, ending with it."""
    end = revisit.step
    if recording.offset_in_episode(end) < history - 1:
        raise ValueError(f"revisit at step {end} has fewer than {history} steps in its episode")
    if context_mode == "raw":
        # The raw window holds the ``history`` steps ending with the entering step.
        start = end - history + 1
        return Transcript(
            frames=recording.tiles[start : end + 1], actions=recording.actions[start : end + 1]
        )
    episode_start = end - recording.offset_in_episode(end)
    if context_mode == "compacted_when_full":
        return compact_when_full(
            recording.actions[episode_start : end + 1],
            recording.tiles[episode_start : end + 1],
            history,
        )
    episode_actions = recording.actions[episode_start:end]
    episode_tiles = recording.tiles[episode_start:end]
    if context_mode == "truncated":
        # The truncated window holds as many steps before entry as the compacted walk, so the two
        # contexts cost the same and differ only in which steps they keep. A walk that steps back
        # through rooms can outgrow a short episode prefix, and the window stops at the episode.
        walk_steps = len(plan_walk(episode_actions.cpu().numpy()).source_steps)
        start = end - min(walk_steps, int(episode_actions.shape[0]))
        return Transcript(
            frames=recording.tiles[start : end + 1], actions=recording.actions[start : end + 1]
        )
    # A harness holding the whole episode so far reduces the steps before entry to a sequence that
    # ends at the current room with the action about to be taken. The entering step follows it,
    # so the model's output at the last kept step predicts the entering frame exactly as it does
    # in the raw window.
    if context_mode == "compacted":
        kept = compact(episode_actions, episode_tiles)
    else:
        kept = deduplicate(episode_actions, episode_tiles)
    frames = torch.cat([kept.frames, recording.tiles[end : end + 1]], dim=0)
    actions = torch.cat([kept.actions, recording.actions[end : end + 1]], dim=0)
    return Transcript(frames=frames, actions=actions)


def frame_accuracy(
    model: WorldModel,
    recording: Recording,
    revisit: Revisit,
    history: int,
    device: torch.device,
    shuffle_distant_history: bool = False,
    preserved_observed_steps: int = LOCAL_STEPS,
    context_mode: ContextMode = "raw",
) -> FrameScore:
    """Score the entering room's tiles as predicted from the steps before it."""
    # The model's outputs at the step before the entering step predict the entering frame in one
    # pass, so nothing of that frame is given away.
    end = revisit.step
    context = probe_context(recording, revisit, history, context_mode)
    context_steps = int(context.frames.shape[0])
    if context_steps > model.settings.max_history:
        raise ValueError(
            f"a context of {context_steps} steps exceeds the model's max_history "
            f"{model.settings.max_history}"
        )
    # frames: (1, steps, 16, 16) int64. actions: (1, steps). A batch of one context.
    frames = model_frames(context.frames)[None].to(device).long()
    actions = context.actions[None].to(device)
    if shuffle_distant_history:
        if recording.episodes < 2:
            raise ValueError("shuffling distant history needs at least two episodes")
        # Another episode's raw steps replace every step before the local path, so the target and
        # the route into it survive while the distant evidence changes.
        episode_index = end // recording.steps_per_episode
        donor_episode = (episode_index + 1) % recording.episodes
        # Windows borrow the donor's steps at their own offsets. Contexts that stand for the
        # whole episode borrow from the donor's start.
        donor_offset = 0
        if context_mode in ("raw", "truncated"):
            donor_offset = (end - context_steps + 1) % recording.steps_per_episode
        donor_start = donor_episode * recording.steps_per_episode + donor_offset
        shuffled_steps = context_steps - 1 - preserved_observed_steps
        if shuffled_steps > 0:
            frames = frames.clone()
            actions = actions.clone()
            donor_end = donor_start + shuffled_steps
            donor_frames = model_frames(recording.tiles[donor_start:donor_end])
            frames[0, :shuffled_steps] = donor_frames.to(device).long()
            actions[0, :shuffled_steps] = recording.actions[donor_start:donor_end].to(device)
    with torch.no_grad():
        # logits: (1, steps, 8, 256), one spatial occupancy score per tile type.
        logits = model(frames, actions)
    # The entering step is the last in the context. The 256 tile positions of the step before
    # it carry the prediction of its tiles across the eight type channels.
    last = context_steps - 1
    # predicted: (256,) the most likely id at each tile position.
    previous_tiles = logits[0, last - 1]
    predicted = previous_tiles.argmax(dim=0)
    # truth: (256,) the frame the model saw as input, with the character drawn in.
    truth = frames[0, last].reshape(TILES_PER_FRAME)
    # background: (256,) the room without the character. The cell the character stands on
    # differs between the two and is skipped, because the room is the ground truth.
    background = recording.tiles[end].reshape(-1).to(device).long()
    # interior: (256,) bool, False on the wall border, whose 56 wall cells and 4 door cells
    # carry nothing that separates one room from another.
    interior = torch.from_numpy(INTERIOR.reshape(-1)).to(device)
    scored = (truth == background) & interior
    accuracy = float((predicted[scored] == truth[scored]).float().mean())
    return FrameScore(accuracy=accuracy, context_steps=context_steps)


def bin_label(gap: int, bins: tuple[int, ...]) -> str:
    validate_gap_bins(bins)
    for low, high in pairwise(bins):
        if low <= gap < high:
            return f"{low}-{high - 1}"
    return f"{bins[-1]}+"


def score_revisits(
    model: WorldModel,
    recording: Recording,
    recording_dir: Path,
    history: int,
    bins: tuple[int, ...],
    device: torch.device,
    limit: int | None = None,
    per_bin: int | None = 40,
    context_mode: ContextMode = "raw",
) -> ProbeResult:
    """Score revisits in a recording and group the accuracies by gap bin."""
    # ``per_bin`` caps how many revisits of each gap bin are scored, drawn at fixed random,
    # since each score is a forward pass over a whole window. 40 per bin keeps
    # a probe at minutes.
    model.eval()
    archive = np.load(recording_dir / "steps.npz")
    revisits = find_revisits(archive["episode"], archive["room"])
    # Every window holds exactly ``history`` steps of one episode, the length the model trained
    # on and the one shape the compiled attention kernel is built for, so revisits in the first
    # steps of an episode are left out.
    revisits = [
        revisit for revisit in revisits if recording.offset_in_episode(revisit.step) >= history - 1
    ]
    if limit is not None:
        revisits = revisits[:limit]
    if per_bin is not None:
        by_bin: dict[str, list[Revisit]] = {}
        for revisit in revisits:
            label = bin_label(revisit.gap, bins)
            by_bin.setdefault(label, []).append(revisit)
        rng = random.Random(42)
        revisits = []
        for members in by_bin.values():
            rng.shuffle(members)
            revisits.extend(members[:per_bin])
        revisits.sort(key=lambda revisit: revisit.step)
    if not revisits:
        raise ValueError(
            f"recording has no revisits with a complete {history}-step history to score"
        )
    scores: list[RevisitScore] = []
    preserved_observed_steps = model.local_receptive_steps()
    for revisit in revisits:
        intact = frame_accuracy(
            model, recording, revisit, history, device, context_mode=context_mode
        )
        shuffled = frame_accuracy(
            model,
            recording,
            revisit,
            history,
            device,
            shuffle_distant_history=True,
            preserved_observed_steps=preserved_observed_steps,
            context_mode=context_mode,
        )
        scores.append(
            RevisitScore(
                step=revisit.step,
                gap=revisit.gap,
                accuracy=intact.accuracy,
                shuffled_accuracy=shuffled.accuracy,
                retrieval_gain=intact.accuracy - shuffled.accuracy,
                # Only a window can miss the target, when the gap is longer than the window.
                history_available=context_mode
                in ("compacted", "deduplicated", "compacted_when_full")
                or revisit.gap < intact.context_steps,
                context_steps=intact.context_steps,
            )
        )
    table: dict[str, list[RevisitScore]] = {}
    for score in scores:
        table.setdefault(bin_label(score.gap, bins), []).append(score)
    report = {
        label: BinSummary(
            mean=float(np.mean([score.accuracy for score in members])),
            shuffled_mean=float(np.mean([score.shuffled_accuracy for score in members])),
            gain_mean=float(np.mean([score.retrieval_gain for score in members])),
            history_available_fraction=float(
                np.mean([score.history_available for score in members])
            ),
            n=len(members),
        )
        for label, members in table.items()
    }
    return ProbeResult(report=report, scores=scores)


def main(settings: ProbeSettings) -> None:
    device = (
        torch.device(settings.device)
        if settings.device
        else torch.accelerator.current_accelerator()
    )
    if device is None:
        raise RuntimeError("no accelerator found. Pass --device cpu to run on the CPU")
    training_identity = recording_identity(Path(settings.training_recording))
    evaluation_identity = recording_identity(Path(settings.recording))
    validate_matching_worlds(training_identity, evaluation_identity)
    recording = load_recording(SplitSettings(recording=settings.recording))
    model = WorldModel(settings.model).to(device)
    state = torch.load(settings.checkpoint, map_location=device)
    checkpoint_config = state.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint has no validated training config")
    checkpoint_model = ModelSettings.model_validate(checkpoint_config.get("model"))
    if checkpoint_model != settings.model:
        raise ValueError("checkpoint model settings do not match the probe")
    if checkpoint_config.get("history") != settings.history:
        raise ValueError("checkpoint history does not match the probe")
    checkpoint_recording = state.get("recording")
    if not isinstance(checkpoint_recording, dict):
        raise ValueError("checkpoint has no recording identity")
    if checkpoint_recording.get("sha256") != training_identity.sha256:
        raise ValueError("checkpoint training recording does not match the probe")
    checkpoint_probe = state.get("probe_recording")
    if checkpoint_probe is not None:
        if not isinstance(checkpoint_probe, dict):
            raise ValueError("checkpoint probe recording identity is malformed")
        if checkpoint_probe.get("sha256") != evaluation_identity.sha256:
            raise ValueError("checkpoint evaluation recording does not match the probe")
    model.load_state_dict(state["model"])
    result = score_revisits(
        model,
        recording,
        Path(settings.recording),
        settings.history,
        settings.bins,
        device,
        settings.limit,
        settings.per_bin,
        settings.context_mode,
    )
    settings.out.mkdir(parents=True, exist_ok=True)
    (settings.out / "probe.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(json.dumps(result.model_dump(mode="json")["report"], indent=2))


if __name__ == "__main__":
    main(tyro.cli(ProbeSettings))
