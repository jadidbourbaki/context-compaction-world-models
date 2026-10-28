"""A tile game whose rooms are generated fresh for every episode, and its recorder."""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tyro
from pydantic import BaseModel, Field, model_validator

# Tile ids. The model reads these numbers directly, so the game needs no learned tokenizer.
FLOOR = 0
GRASS = 1
WALL = 2
DOOR = 3
ROCK = 4
TREE = 5
WATER = 6
CHARACTER = 7
TILE_VOCAB = 8

# Tiles a patch of scenery can be made of, drawn from uniformly. Grass is decoration, and it
# appears twice so that two patches in five leave the room open.
SCENERY = (GRASS, GRASS, ROCK, TREE, WATER)
# Each tile id selects one 16 by 16 sprite from the CC0 Arduboy graphics collection.
SPRITE_SIZE = 16
SPRITE_SHEET = Path(__file__).parent / "assets" / "arduboy_graphics_v1.png"
# Coordinates are the left and top pixel of each sprite in tile-id order.
SPRITE_COORDINATES = np.array(
    [
        [48, 224],
        [16, 224],
        [128, 32],
        [144, 64],
        [0, 224],
        [64, 32],
        [0, 176],
        [16, 16],
    ],
    dtype=np.int64,
)

# One room is a screen. The model's frame size follows these two numbers.
ROOM_ROWS = 16
ROOM_COLS = 16
# 56 of the 60 border cells are wall in every room and the other 4 hold that room's doors, so
# the probe scores the 196 cells inside the border, where two rooms actually differ.
INTERIOR = np.pad(np.ones((ROOM_ROWS - 2, ROOM_COLS - 2), dtype=bool), 1)
# The interior without the centre cell, where the character stands in every frame, so the 195
# cells a room prediction is scored on.
SCORED_CELLS = INTERIOR.copy()
SCORED_CELLS[ROOM_ROWS // 2, ROOM_COLS // 2] = False

# An episode's id carries the seed that generated it, so two recordings made with different
# seeds never name the same room and an evaluation room is always one training never saw.
EPISODES_PER_SEED = 1_000_000
# A room's key packs its index into one byte beside its episode's id, so a world holds at most
# this many rooms. Beyond it two episodes would share a key and a revisit could join two worlds.
ROOMS_PER_WORLD = 256
# Version 3 records one complete room transition and derives the centred character for model input.
RECORDING_FORMAT_VERSION = 3

ACTIONS = ("up", "down", "left", "right")
# The row and column each action adds to the character's position, in the order of ACTIONS.
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def room_keys(episode: np.ndarray, room: np.ndarray) -> np.ndarray:
    """One number per generated room, which no other episode of any recording shares."""
    return episode.astype(np.int64) * ROOMS_PER_WORLD + room.astype(np.int64)


class WorldSettings(BaseModel):
    """Everything that decides a recording, so a recording is reproducible from its meta.json."""

    out: Path = Field(description="directory to write the recording and preview video into")
    episodes: int = Field(default=1000, ge=1, description="one generated world per episode")
    steps_per_episode: int = Field(default=512, ge=2)
    rooms_across: int = Field(default=8, ge=2)
    rooms_down: int = Field(default=8, ge=2)
    # A room holds this many patches of scenery, each a square of up to this radius.
    patches_per_room: int = Field(default=28, ge=0)
    patch_radius: int = Field(default=1, ge=1)
    seed: int = Field(default=42)
    video_steps: int = Field(default=64, ge=0, description="room transitions written to steps.gif")

    @model_validator(mode="after")
    def rooms_fit_one_key(self) -> WorldSettings:
        rooms = self.rooms_across * self.rooms_down
        if rooms > ROOMS_PER_WORLD:
            raise ValueError(f"{rooms} rooms exceed the {ROOMS_PER_WORLD} a room key can hold")
        if self.episodes > EPISODES_PER_SEED:
            raise ValueError(f"{self.episodes} episodes exceed the {EPISODES_PER_SEED} per seed")
        return self


class Doors(BaseModel):
    """Where the door on each side of a room sits, or None where the world ends."""

    # north and south hold a column. west and east hold a row.
    north: int | None
    south: int | None
    west: int | None
    east: int | None


class Cell(BaseModel):
    """A position inside a room."""

    row: int
    col: int


class Doorway(BaseModel):
    """A door cell on the border of one room."""

    row: int
    col: int

    def cell(self) -> Cell:
        return Cell(row=self.row, col=self.col)


class World(BaseModel):
    """The rooms of one episode and the door positions that join them."""

    model_config = {"arbitrary_types_allowed": True}

    # tiles: (rooms_down, rooms_across, ROOM_ROWS, ROOM_COLS) uint8 tile ids.
    tiles: np.ndarray
    # east_door_row[r][c] is the row of the door between room (r, c) and room (r, c + 1).
    east_door_row: np.ndarray
    # south_door_col[r][c] is the column of the door between room (r, c) and room (r + 1, c).
    south_door_col: np.ndarray


def doorways(doors: Doors) -> list[Doorway]:
    """The border cell occupied by each door."""
    found = []
    if doors.north is not None:
        found.append(Doorway(row=0, col=doors.north))
    if doors.south is not None:
        found.append(Doorway(row=ROOM_ROWS - 1, col=doors.south))
    if doors.west is not None:
        found.append(Doorway(row=doors.west, col=0))
    if doors.east is not None:
        found.append(Doorway(row=doors.east, col=ROOM_COLS - 1))
    return found


def carve(tiles: np.ndarray, start: Cell, end: Cell) -> None:
    """Clear an L-shaped path of floor between two cells, staying inside the room's border."""
    # A door sits on the border, so the path starts one cell inside it.
    row = int(np.clip(start.row, 1, ROOM_ROWS - 2))
    col = int(np.clip(start.col, 1, ROOM_COLS - 2))
    tiles[row, min(col, end.col) : max(col, end.col) + 1] = FLOOR
    tiles[min(row, end.row) : max(row, end.row) + 1, end.col] = FLOOR


def generate_room(rng: np.random.Generator, settings: WorldSettings, doors: Doors) -> np.ndarray:
    """One room as a (ROOM_ROWS, ROOM_COLS) uint8 grid of tile ids."""
    tiles = np.full((ROOM_ROWS, ROOM_COLS), FLOOR, dtype=np.uint8)
    # Scenery arrives as square patches, so a room holds shapes a model can learn to expect
    # while the arrangement of a particular room stays unpredictable.
    for _ in range(settings.patches_per_room):
        tile = int(rng.choice(SCENERY))
        centre_row = int(rng.integers(1, ROOM_ROWS - 1))
        centre_col = int(rng.integers(1, ROOM_COLS - 1))
        radius = int(rng.integers(1, settings.patch_radius + 1))
        top = max(1, centre_row - radius)
        bottom = min(ROOM_ROWS - 1, centre_row + radius + 1)
        left = max(1, centre_col - radius)
        right = min(ROOM_COLS - 1, centre_col + radius + 1)
        tiles[top:bottom, left:right] = tile
    # A corridor from every door to the middle of the room joins all the doors to each other,
    # so a patch of water can never seal the character into a corner.
    middle = Cell(row=ROOM_ROWS // 2, col=ROOM_COLS // 2)
    for doorway in doorways(doors):
        carve(tiles, doorway.cell(), middle)
    tiles[0, :] = WALL
    tiles[ROOM_ROWS - 1, :] = WALL
    tiles[:, 0] = WALL
    tiles[:, ROOM_COLS - 1] = WALL
    for doorway in doorways(doors):
        tiles[doorway.row, doorway.col] = DOOR
    return tiles


def room_doors(settings: WorldSettings, world: World, room_row: int, room_col: int) -> Doors:
    """The doors of one room, taken from the door positions shared with its neighbours."""
    north = None if room_row == 0 else int(world.south_door_col[room_row - 1, room_col])
    south = (
        None
        if room_row == settings.rooms_down - 1
        else int(world.south_door_col[room_row, room_col])
    )
    west = None if room_col == 0 else int(world.east_door_row[room_row, room_col - 1])
    east = (
        None
        if room_col == settings.rooms_across - 1
        else int(world.east_door_row[room_row, room_col])
    )
    return Doors(north=north, south=south, west=west, east=east)


def generate_world(rng: np.random.Generator, settings: WorldSettings) -> World:
    """A fresh grid of connected rooms for one episode."""
    # Every pair of neighbouring rooms shares one door, drawn away from the corners so the
    # corridor to the middle of the room always has room to turn. Both arrays are
    # (rooms_down, rooms_across) int64, one door position per room's south and east side.
    east_door_row = rng.integers(
        2, ROOM_ROWS - 2, size=(settings.rooms_down, settings.rooms_across)
    )
    south_door_col = rng.integers(
        2, ROOM_COLS - 2, size=(settings.rooms_down, settings.rooms_across)
    )
    # tiles: (rooms_down, rooms_across, 16, 16) uint8, filled one room at a time below.
    tiles = np.zeros(
        (settings.rooms_down, settings.rooms_across, ROOM_ROWS, ROOM_COLS), dtype=np.uint8
    )
    world = World(tiles=tiles, east_door_row=east_door_row, south_door_col=south_door_col)
    for room_row in range(settings.rooms_down):
        for room_col in range(settings.rooms_across):
            doors = room_doors(settings, world, room_row, room_col)
            world.tiles[room_row, room_col] = generate_room(rng, settings, doors)
    return world


def valid_room_actions(settings: WorldSettings, room: int) -> list[int]:
    """Cardinal actions whose destination lies inside the room grid."""
    room_row = room // settings.rooms_across
    room_col = room % settings.rooms_across
    actions = []
    for action, move in enumerate(MOVES):
        next_row = room_row + move[0]
        next_col = room_col + move[1]
        if 0 <= next_row < settings.rooms_down and 0 <= next_col < settings.rooms_across:
            actions.append(action)
    return actions


def move_room(settings: WorldSettings, room: int, action: int) -> int:
    """Apply one cardinal action to a room and return its neighbouring room."""
    if action not in valid_room_actions(settings, room):
        raise ValueError(f"action {action} is invalid from room {room}")
    room_row = room // settings.rooms_across
    room_col = room % settings.rooms_across
    move = MOVES[action]
    next_row = room_row + move[0]
    next_col = room_col + move[1]
    return next_row * settings.rooms_across + next_col


class RoomWalker:
    """Choose valid room transitions while avoiding an immediate reversal when possible."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.previous_room: int | None = None

    def choose(self, settings: WorldSettings, room: int) -> int:
        """Choose one valid cardinal action from the current room."""
        actions = valid_room_actions(settings, room)
        forward_actions = [
            action
            for action in actions
            if self.previous_room is None or move_room(settings, room, action) != self.previous_room
        ]
        if forward_actions:
            actions = forward_actions
        action = actions[int(self.rng.integers(len(actions)))]
        self.previous_room = room
        return action


class Steps(BaseModel):
    """The arrays one call to record() produces, one row per step."""

    model_config = {"arbitrary_types_allowed": True}

    # tiles: (steps, 16, 16) uint8 logical tile ids without the character.
    tiles: np.ndarray
    actions: np.ndarray
    # episode holds the world's id and room its index in that world's grid, so
    # episode * ROOMS_PER_WORLD + room names one generated room across every recording.
    episode: np.ndarray
    room: np.ndarray


def record(settings: WorldSettings) -> Steps:
    """Navigate one fresh room grid per episode and record every room transition."""
    rng = np.random.default_rng(settings.seed)
    total = settings.episodes * settings.steps_per_episode
    tiles = np.zeros((total, ROOM_ROWS, ROOM_COLS), dtype=np.uint8)
    actions = np.zeros(total, dtype=np.uint8)
    episode_ids = np.zeros(total, dtype=np.int64)
    rooms = np.zeros(total, dtype=np.int64)
    for episode in range(settings.episodes):
        episode_id = settings.seed * EPISODES_PER_SEED + episode
        world = generate_world(rng, settings)
        room_count = settings.rooms_across * settings.rooms_down
        room = int(rng.integers(room_count))
        walker = RoomWalker(rng)
        for offset in range(settings.steps_per_episode):
            written = episode * settings.steps_per_episode + offset
            room_row = room // settings.rooms_across
            room_col = room % settings.rooms_across
            room_tiles = world.tiles[room_row, room_col]
            tiles[written] = room_tiles
            action = walker.choose(settings, room)
            actions[written] = action
            episode_ids[written] = episode_id
            rooms[written] = room
            room = move_room(settings, room, action)
        if episode % 100 == 0:
            logging.getLogger(__name__).info("episode %d of %d", episode, settings.episodes)
    return Steps(tiles=tiles, actions=actions, episode=episode_ids, room=rooms)


def add_character(tiles: np.ndarray) -> np.ndarray:
    """Copy logical room tiles and place the character at the centre of every frame."""
    screen = tiles.copy()
    screen[..., ROOM_ROWS // 2, ROOM_COLS // 2] = CHARACTER
    return screen


@cache
def sprite_atlas() -> np.ndarray:
    """Load the eight selected 16 by 16 sprites from the source sheet."""
    source = imageio.imread(SPRITE_SHEET)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"sprite sheet has shape {source.shape}, expected RGB pixels")
    # sprites: (8, 16, 16, 3) uint8, one CC0 sprite for each logical tile id.
    sprites = np.zeros((TILE_VOCAB, SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8)
    for tile_id in range(TILE_VOCAB):
        left = int(SPRITE_COORDINATES[tile_id, 0])
        top = int(SPRITE_COORDINATES[tile_id, 1])
        right = left + SPRITE_SIZE
        bottom = top + SPRITE_SIZE
        sprite = source[top:bottom, left:right]
        if sprite.shape != (SPRITE_SIZE, SPRITE_SIZE, 3):
            raise ValueError(f"sprite {tile_id} has shape {sprite.shape}")
        sprites[tile_id] = sprite
    return sprites


def render_room(screen: np.ndarray) -> np.ndarray:
    """Render one room of tile ids as sprites inside a one-sprite black border."""
    # arranged: (16, 16, 16, 16, 3) uint8, with sprite rows beside room rows.
    sprites = sprite_atlas()
    arranged = sprites[screen].copy()
    horizontal_wall = sprites[WALL]
    vertical_wall = np.fliplr(np.rot90(horizontal_wall, -1))
    upward_door = sprites[DOOR]
    downward_door = np.rot90(upward_door, 2)
    leftward_door = np.rot90(upward_door)
    rightward_door = np.rot90(upward_door, -1)
    for col in range(ROOM_COLS):
        arranged[0, col] = upward_door if screen[0, col] == DOOR else horizontal_wall
        arranged[ROOM_ROWS - 1, col] = (
            downward_door if screen[ROOM_ROWS - 1, col] == DOOR else horizontal_wall
        )
    for row in range(1, ROOM_ROWS - 1):
        arranged[row, 0] = leftward_door if screen[row, 0] == DOOR else vertical_wall
        arranged[row, ROOM_COLS - 1] = (
            rightward_door if screen[row, ROOM_COLS - 1] == DOOR else vertical_wall
        )
    room_pixels = arranged.transpose(0, 2, 1, 3, 4)
    room_pixels = room_pixels.reshape(ROOM_ROWS * SPRITE_SIZE, ROOM_COLS * SPRITE_SIZE, 3)
    # bordered: (288, 288, 3) uint8, with one black sprite cell around the room.
    bordered = np.zeros(
        ((ROOM_ROWS + 2) * SPRITE_SIZE, (ROOM_COLS + 2) * SPRITE_SIZE, 3),
        dtype=np.uint8,
    )
    bordered[SPRITE_SIZE:-SPRITE_SIZE, SPRITE_SIZE:-SPRITE_SIZE] = room_pixels
    return bordered


def preview_frame(screen: np.ndarray, action: int) -> np.ndarray:
    """Render one room with an outer marker showing the selected cardinal action."""
    # bordered: (288, 288, 3) uint8, the room inside a black band that holds the marker.
    bordered = render_room(screen)
    if action == 0:
        door_positions = np.flatnonzero(screen[0] == DOOR)
        indicator_row = 0
    elif action == 1:
        door_positions = np.flatnonzero(screen[ROOM_ROWS - 1] == DOOR)
        indicator_row = ROOM_ROWS + 1
    elif action == 2:
        door_positions = np.flatnonzero(screen[:, 0] == DOOR)
        indicator_col = 0
    elif action == 3:
        door_positions = np.flatnonzero(screen[:, ROOM_COLS - 1] == DOOR)
        indicator_col = ROOM_COLS + 1
    else:
        raise ValueError(f"action {action} is outside 0 through {len(ACTIONS) - 1}")
    if door_positions.shape != (1,):
        raise ValueError(f"action {action} needs exactly one door, found {len(door_positions)}")
    door_position = int(door_positions[0]) + 1
    if action < 2:
        indicator_col = door_position
    else:
        indicator_row = door_position
    indicator_top = indicator_row * SPRITE_SIZE
    indicator_left = indicator_col * SPRITE_SIZE
    bordered[
        indicator_top : indicator_top + SPRITE_SIZE,
        indicator_left : indicator_left + SPRITE_SIZE,
    ] = 255
    return bordered


def write_preview(path: Path, tiles: np.ndarray, actions: np.ndarray, steps: int) -> None:
    """Render the first steps of a recording as an animated GIF, one sprite per tile."""
    if steps == 0:
        return
    # Four room transitions per second leaves enough time to inspect each generated layout.
    writer = imageio.get_writer(path, mode="I", duration=250, loop=0)
    screen = add_character(tiles[:steps])
    for frame, action in zip(screen, actions[:steps], strict=True):
        writer.append_data(preview_frame(frame, int(action)))
    writer.close()


def main(settings: WorldSettings) -> None:
    logger = logging.getLogger(__name__)
    recording = record(settings)
    settings.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        settings.out / "steps.npz",
        tiles=recording.tiles,
        actions=recording.actions,
        episode=recording.episode,
        room=recording.room,
    )
    write_preview(
        settings.out / "steps.gif",
        recording.tiles,
        recording.actions,
        settings.video_steps,
    )
    meta = {
        "format_version": RECORDING_FORMAT_VERSION,
        "settings": settings.model_dump(mode="json"),
        "steps": int(recording.actions.shape[0]),
        "rooms": int(np.unique(room_keys(recording.episode, recording.room)).shape[0]),
    }
    (settings.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("wrote %s", settings.out / "steps.npz")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(WorldSettings))
