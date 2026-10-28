"""Turn a recording into training windows and say how each room relates to the training data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, DirectoryPath, Field

from shared_cache_world_model.world import (
    ACTIONS,
    CHARACTER,
    EPISODES_PER_SEED,
    MOVES,
    RECORDING_FORMAT_VERSION,
    ROOM_COLS,
    ROOM_ROWS,
    TILE_VOCAB,
    WorldSettings,
    room_keys,
)

# One frame is one room, so the model's frame size is the room size the game generates.
TILES_PER_FRAME = ROOM_ROWS * ROOM_COLS


class SplitSettings(BaseModel):
    recording: DirectoryPath = Field(description="directory holding steps.npz from the game")


class RecordingIdentity(BaseModel):
    """The generated settings and content hash that identify one recording."""

    format_version: Literal[3]
    sha256: str
    settings: WorldSettings
    steps: int
    rooms: int


class RecordingMeta(BaseModel):
    """The validated contents of a generated recording's meta.json."""

    model_config = {"extra": "forbid"}

    format_version: Literal[3]
    settings: WorldSettings
    steps: int = Field(ge=1)
    rooms: int = Field(ge=1)


def recording_identity(recording_dir: Path) -> RecordingIdentity:
    """Validate a generated recording's metadata and hash its arrays."""
    meta_path = recording_dir / "meta.json"
    archive_path = recording_dir / "steps.npz"
    metadata = RecordingMeta.model_validate(json.loads(meta_path.read_text()))
    with np.load(archive_path) as archive:
        required = {"tiles", "actions", "episode", "room"}
        missing = sorted(required - set(archive.files))
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"steps.npz is missing arrays: {joined}")
        unexpected = sorted(set(archive.files) - required)
        if unexpected:
            joined = ", ".join(unexpected)
            raise ValueError(f"steps.npz has unexpected arrays: {joined}")
        actual_steps = int(archive["actions"].shape[0])
        expected_shapes = {
            "tiles": (actual_steps, ROOM_ROWS, ROOM_COLS),
            "actions": (actual_steps,),
            "episode": (actual_steps,),
            "room": (actual_steps,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = archive[name].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"steps.npz array {name} has shape {actual_shape}, expected {expected_shape}"
                )
        expected_dtypes = {
            "tiles": np.dtype(np.uint8),
            "actions": np.dtype(np.uint8),
            "episode": np.dtype(np.int64),
            "room": np.dtype(np.int64),
        }
        for name, expected_dtype in expected_dtypes.items():
            actual_dtype = archive[name].dtype
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"steps.npz array {name} has dtype {actual_dtype}, expected {expected_dtype}"
                )
        expected_steps = metadata.settings.episodes * metadata.settings.steps_per_episode
        if metadata.steps != expected_steps:
            raise ValueError(
                f"meta.json reports {metadata.steps} steps, settings require {expected_steps}"
            )
        if np.any(archive["actions"] >= len(ACTIONS)):
            raise ValueError(f"steps.npz has an action outside 0 through {len(ACTIONS) - 1}")
        if np.any(archive["tiles"] >= TILE_VOCAB):
            raise ValueError(f"steps.npz has a tile outside 0 through {TILE_VOCAB - 1}")
        if np.any(archive["tiles"] == CHARACTER):
            raise ValueError("steps.npz room tiles contain the character token")
        first_episode = metadata.settings.seed * EPISODES_PER_SEED
        expected_episodes = np.arange(
            first_episode,
            first_episode + metadata.settings.episodes,
            dtype=np.int64,
        )
        expected_episode_rows = np.repeat(
            expected_episodes,
            metadata.settings.steps_per_episode,
        )
        if not np.array_equal(archive["episode"], expected_episode_rows):
            raise ValueError("steps.npz episode ids do not match meta.json settings")
        room_limit = metadata.settings.rooms_across * metadata.settings.rooms_down
        if np.any(archive["room"] < 0) or np.any(archive["room"] >= room_limit):
            raise ValueError(f"steps.npz has a room outside 0 through {room_limit - 1}")
        # Every action selects a valid neighbouring room, including the final unobserved move.
        room_rows = archive["room"] // metadata.settings.rooms_across
        room_cols = archive["room"] % metadata.settings.rooms_across
        moves = np.asarray(MOVES)
        chosen_moves = moves[archive["actions"]]
        next_rows = room_rows + chosen_moves[:, 0]
        next_cols = room_cols + chosen_moves[:, 1]
        valid_rows = (next_rows >= 0) & (next_rows < metadata.settings.rooms_down)
        valid_cols = (next_cols >= 0) & (next_cols < metadata.settings.rooms_across)
        if not np.all(valid_rows & valid_cols):
            raise ValueError("steps.npz has an action that leaves the room grid")
        expected_rooms = next_rows * metadata.settings.rooms_across + next_cols
        same_episode = archive["episode"][1:] == archive["episode"][:-1]
        if not np.array_equal(expected_rooms[:-1][same_episode], archive["room"][1:][same_episode]):
            raise ValueError("steps.npz room transitions do not match its actions")
        keys = room_keys(archive["episode"], archive["room"])
        actual_rooms = int(np.unique(keys).shape[0])
    if metadata.steps != actual_steps:
        raise ValueError(f"meta.json reports {metadata.steps} steps, archive has {actual_steps}")
    if metadata.rooms != actual_rooms:
        raise ValueError(f"meta.json reports {metadata.rooms} rooms, archive has {actual_rooms}")
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        chunk = stream.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = stream.read(1024 * 1024)
    return RecordingIdentity(
        format_version=RECORDING_FORMAT_VERSION,
        sha256=digest.hexdigest(),
        settings=metadata.settings,
        steps=metadata.steps,
        rooms=metadata.rooms,
    )


def validate_matching_worlds(training: RecordingIdentity, evaluation: RecordingIdentity) -> None:
    """Require training and evaluation recordings to use the same world distribution."""
    fields = (
        "steps_per_episode",
        "rooms_across",
        "rooms_down",
        "patches_per_room",
        "patch_radius",
    )
    mismatched = [
        field
        for field in fields
        if getattr(training.settings, field) != getattr(evaluation.settings, field)
    ]
    if mismatched:
        joined = ", ".join(mismatched)
        raise ValueError(f"training and evaluation world settings differ: {joined}")
    if training.settings.seed == evaluation.settings.seed:
        raise ValueError(f"training and evaluation recordings share seed {training.settings.seed}")


class Recording(BaseModel):
    """The arrays of one recording as tensors."""

    model_config = {"arbitrary_types_allowed": True}

    # tiles: (steps, 16, 16) uint8 room tile ids without the character, the probe's ground truth.
    # The frames stay 8 bit in memory, a quarter of what int64 would take over millions of
    # steps, and a gathered batch is cast where the model needs it.
    tiles: torch.Tensor
    # actions: (steps,) int64 indices into ACTIONS.
    actions: torch.Tensor
    # rooms: (steps,) int64 room keys, one per generated room.
    rooms: torch.Tensor
    # Every episode is one generated world. A window never spans two of them.
    steps_per_episode: int

    @property
    def steps(self) -> int:
        return int(self.tiles.shape[0])

    @property
    def episodes(self) -> int:
        return self.steps // self.steps_per_episode

    def offset_in_episode(self, step: int) -> int:
        """How many steps of that step's episode came before it."""
        return step % self.steps_per_episode


def episode_length(episode: np.ndarray) -> int:
    """The number of steps in every episode, which must be the same for all of them."""
    counts = np.unique(episode, return_counts=True)[1]
    if counts.min() != counts.max():
        raise ValueError(f"episodes run from {counts.min()} to {counts.max()} steps, expected one")
    return int(counts[0])


def load_recording(settings: SplitSettings) -> Recording:
    with np.load(Path(settings.recording) / "steps.npz") as archive:
        if "episode" not in archive:
            raise ValueError(f"{settings.recording}/steps.npz has no episode array")
        # keys: (steps,) int64. An episode's id carries its seed, so the same room index in two
        # recordings names two different generated rooms.
        keys = room_keys(archive["episode"], archive["room"])
        rooms = torch.from_numpy(keys)
        steps_per_episode = episode_length(archive["episode"])
        tiles = torch.from_numpy(archive["tiles"])
        actions = torch.from_numpy(archive["actions"].astype(np.int64))
    return Recording(
        tiles=tiles,
        actions=actions,
        rooms=rooms,
        steps_per_episode=steps_per_episode,
    )


def model_frames(tiles: torch.Tensor) -> torch.Tensor:
    """Copy room tiles and place the model's character token at the centre of every frame."""
    frames = tiles.clone()
    frames[..., ROOM_ROWS // 2, ROOM_COLS // 2] = CHARACTER
    return frames


class Windows(BaseModel):
    """One batch of training windows, each ``history`` consecutive steps of the recording."""

    model_config = {"arbitrary_types_allowed": True}

    # frames: (batch, history, 16, 16) int64 tile ids with the character drawn in.
    frames: torch.Tensor
    # actions: (batch, history) int64 indices into ACTIONS.
    actions: torch.Tensor
    # revisiting: (batch, history) bool, True when the room occurred earlier in the window.
    revisiting: torch.Tensor


def window_revisits(rooms: torch.Tensor) -> torch.Tensor:
    """Mark each room that also occurred at an earlier position of its window."""
    history = rooms.shape[1]
    # same_room: (batch, history, history) bool, indexed by current and candidate earlier step.
    same_room = rooms[:, :, None] == rooms[:, None, :]
    # earlier: (history, history) bool, True when the column precedes the row.
    positions = torch.arange(history, device=rooms.device)
    earlier = positions[None, :] < positions[:, None]
    return (same_room & earlier[None]).any(dim=2)


def sample_windows(
    recording: Recording, history: int, batch_size: int, generator: torch.Generator
) -> Windows:
    """Random training windows of ``history`` consecutive steps, each inside one episode."""
    if recording.steps_per_episode < history:
        raise ValueError(
            f"episodes of {recording.steps_per_episode} steps are shorter than history {history}"
        )
    # A window is drawn as an episode and an offset inside it, so it never spans two worlds.
    last_offset = recording.steps_per_episode - history
    episodes = torch.randint(0, recording.episodes, (batch_size,), generator=generator)
    starting_offsets = torch.randint(0, last_offset + 1, (batch_size,), generator=generator)
    # starts: (batch,) int64 first steps of each window.
    starts = episodes * recording.steps_per_episode + starting_offsets
    # window_steps: (batch, history) int64, the step index at every window position.
    window_steps = starts[:, None] + torch.arange(history)[None, :]
    # Indexing a (steps, ...) array with window_steps gathers a (batch, history, ...) array.
    # frames: (batch, history, 16, 16) int64, the dtype the embedding lookup takes.
    frames = model_frames(recording.tiles[window_steps]).long()
    # actions: (batch, history) int64.
    actions = recording.actions[window_steps]
    # rooms: (batch, history) int64 generated-room keys.
    rooms = recording.rooms[window_steps]
    # revisiting: (batch, history) bool, based only on history available inside each window.
    revisiting = window_revisits(rooms)
    return Windows(frames=frames, actions=actions, revisiting=revisiting)
