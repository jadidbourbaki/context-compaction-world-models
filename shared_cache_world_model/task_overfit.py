"""Prove both cache designs can retrieve a generated room through the main decoder."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import tyro
from pydantic import BaseModel, Field

from shared_cache_world_model.data import model_frames
from shared_cache_world_model.model import (
    Kind,
    ModelSettings,
    WorldModel,
    masked_mean,
    next_frame_losses,
    tile_target_mask,
)
from shared_cache_world_model.train import current_commit
from shared_cache_world_model.world import (
    ACTIONS,
    INTERIOR,
    ROOM_COLS,
    ROOM_ROWS,
    Doors,
    WorldSettings,
    generate_room,
)

SEQUENCE_STEPS = 7
SOURCE_STEP = 0
QUERY_STEP = SEQUENCE_STEPS - 2
TARGET_STEP = SEQUENCE_STEPS - 1
KINDS: tuple[Kind, ...] = ("per_layer", "shared_cache", "ced", "ced_full")
logger = logging.getLogger(__name__)


class TaskOverfitSettings(BaseModel):
    seed: int = 42
    room_pairs: int = Field(default=8, ge=1)
    steps: int = Field(default=600, ge=1)
    d_model: int = Field(default=64, ge=8)
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=4, ge=2)
    lr: float = Field(default=3e-3, gt=0.0)
    min_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    min_retrieval_gain: float = Field(default=0.50, ge=0.0, le=1.0)
    device: str = "cpu"
    out: Path | None = None


class TaskScore(BaseModel):
    kind: Kind
    accuracy: float
    shuffled_accuracy: float
    retrieval_gain: float
    final_loss: float
    passed: bool


class TaskReport(BaseModel):
    config: dict[str, object]
    scores: list[TaskScore]
    passed: bool
    commit: str
    versions: dict[str, str]
    device: str
    started: str
    duration_seconds: float


def task_batch(
    room_pairs: int, rng: np.random.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired histories whose identical local query has two possible next rooms."""
    examples = room_pairs * 2
    # tiles: (2 * room_pairs, 7, 16, 16) uint8 logical room tiles.
    tiles = np.zeros((examples, SEQUENCE_STEPS, ROOM_ROWS, ROOM_COLS), dtype=np.uint8)
    # actions: (2 * room_pairs, 7) int64, shared by both variants of each pair.
    actions = np.zeros((examples, SEQUENCE_STEPS), dtype=np.int64)
    settings = WorldSettings(out=Path("unused"), episodes=1, steps_per_episode=2)
    doors = Doors(north=4, south=9, west=6, east=11)
    for pair in range(room_pairs):
        first_example = pair * 2
        second_example = first_example + 1
        first_source = generate_room(rng, settings, doors)
        second_source = generate_room(rng, settings, doors)
        if np.array_equal(first_source, second_source):
            raise ValueError(f"room pair {pair} contains two equal source rooms")
        local_rooms = np.stack(
            [generate_room(rng, settings, doors) for _ in range(QUERY_STEP)],
            axis=0,
        )
        pair_actions = rng.integers(0, len(ACTIONS), size=SEQUENCE_STEPS, dtype=np.int64)
        tiles[first_example, SOURCE_STEP] = first_source
        tiles[second_example, SOURCE_STEP] = second_source
        tiles[first_example, 1 : QUERY_STEP + 1] = local_rooms
        tiles[second_example, 1 : QUERY_STEP + 1] = local_rooms
        tiles[first_example, TARGET_STEP] = first_source
        tiles[second_example, TARGET_STEP] = second_source
        actions[first_example] = pair_actions
        actions[second_example] = pair_actions
    # tile_tensor: (2 * room_pairs, 7, 16, 16) uint8 logical room tiles.
    tile_tensor = torch.from_numpy(tiles)
    # frames: (2 * room_pairs, 7, 16, 16) uint8 model inputs with the character.
    frames = model_frames(tile_tensor).to(device)
    # action_tensor: (2 * room_pairs, 7) int64 action ids.
    action_tensor = torch.from_numpy(actions).to(device)
    return frames, action_tensor


def shuffle_sources(frames: torch.Tensor) -> torch.Tensor:
    """Swap the distant source rooms within each pair and preserve every target."""
    if frames.shape[0] % 2 != 0:
        raise ValueError(f"paired task has odd batch size {frames.shape[0]}")
    shuffled = frames.clone()
    shuffled[0::2, SOURCE_STEP] = frames[1::2, SOURCE_STEP]
    shuffled[1::2, SOURCE_STEP] = frames[0::2, SOURCE_STEP]
    return shuffled


def final_transition_loss(logits: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Apply the main next-frame loss to the final query and its room target."""
    # target_steps: (batch, 7) bool, with only step 6 selected as a prediction target.
    target_steps = torch.zeros(frames.shape[:2], dtype=torch.bool, device=frames.device)
    target_steps[:, TARGET_STEP] = True
    targets = tile_target_mask(target_steps)
    losses = next_frame_losses(logits, frames)
    return masked_mean(losses, targets)


def task_accuracy(model: WorldModel, frames: torch.Tensor, actions: torch.Tensor) -> float:
    """Score informative room cells predicted by the final query step."""
    with torch.no_grad():
        # logits: (batch, 7, 8, 256), including the query prediction at step 5.
        logits = model(frames, actions)
    predicted = logits[:, QUERY_STEP].argmax(dim=1)
    truth = frames[:, TARGET_STEP].reshape(frames.shape[0], -1)
    # scored: (256,) bool. The fixed wall border and character cell carry no room identity.
    scored = torch.from_numpy(INTERIOR.reshape(-1)).to(frames.device).clone()
    scored[ROOM_ROWS // 2 * ROOM_COLS + ROOM_COLS // 2] = False
    correct = predicted[:, scored] == truth[:, scored]
    return float(correct.float().mean())


def train_kind(
    kind: Kind,
    settings: TaskOverfitSettings,
    frames: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> TaskScore:
    """Overfit one cache design on the fixed paired room task."""
    torch.manual_seed(settings.seed)
    model_settings = ModelSettings(
        kind=kind,
        d_model=settings.d_model,
        n_heads=settings.n_heads,
        n_layers=settings.n_layers,
        max_history=SEQUENCE_STEPS,
    )
    model = WorldModel(model_settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=0.0)
    final_loss = float("nan")
    model.train()
    for step in range(settings.steps):
        logits = model(frames, actions)
        loss = final_transition_loss(logits, frames)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if (step + 1) % 25 == 0:
            logger.info("%s step %d loss %.4f", kind, step + 1, final_loss)
    model.eval()
    accuracy = task_accuracy(model, frames, actions)
    shuffled_accuracy = task_accuracy(model, shuffle_sources(frames), actions)
    retrieval_gain = accuracy - shuffled_accuracy
    passed = accuracy >= settings.min_accuracy and retrieval_gain >= settings.min_retrieval_gain
    return TaskScore(
        kind=kind,
        accuracy=accuracy,
        shuffled_accuracy=shuffled_accuracy,
        retrieval_gain=retrieval_gain,
        final_loss=final_loss,
        passed=passed,
    )


def main(settings: TaskOverfitSettings) -> None:
    started = time.time()
    device = torch.device(settings.device)
    rng = np.random.default_rng(settings.seed)
    batch = task_batch(settings.room_pairs, rng, device)
    frames = batch[0]
    actions = batch[1]
    scores = [train_kind(kind, settings, frames, actions, device) for kind in KINDS]
    report = TaskReport(
        config=settings.model_dump(mode="json"),
        scores=scores,
        passed=all(score.passed for score in scores),
        commit=current_commit(),
        versions={"torch": torch.__version__, "numpy": version("numpy")},
        device=str(device),
        started=datetime.fromtimestamp(started, tz=UTC).isoformat(timespec="seconds"),
        duration_seconds=round(time.time() - started, 1),
    )
    rendered = report.model_dump_json(indent=2) + "\n"
    if settings.out is None:
        print(rendered, end="")
    else:
        settings.out.parent.mkdir(parents=True, exist_ok=True)
        settings.out.write_text(rendered)
        logger.info("wrote %s", settings.out)
    if not report.passed:
        raise RuntimeError("paired room retrieval diagnostic did not meet its thresholds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(TaskOverfitSettings))
