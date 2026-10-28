# Data

Our benchmark world generates a new world for every episode. Therefore, we can reproduce each
recording from the settings that made it.

## Recordings

Generate the training recording.

```sh
just world --out data/roomnav_train --episodes 5000 --seed 42
```

Generate the evaluation recording.

```sh
just world --out data/roomnav_eval --episodes 400 --seed 43
```

| Recording | Episodes | Steps | Seed | SHA-256 of `steps.npz` |
|---|---:|---:|---:|---|
| `roomnav_train` | 5,000 | 2,560,000 | 42 | `e1d07fff710162125bbb707ba6c944a271ad18275cbdcdd210b14fa8da1bc667` |
| `roomnav_eval` | 400 | 204,800 | 43 | `2771c1874b46c60592ccc1a296b8d87914b006495489a38fd780c517f18da5ba` |

Every episode has 512 steps. Revisits per gap bin:

| Gap bin | `roomnav_train` | `roomnav_eval` |
|---|---:|---:|
| 1–7 | 432,718 | 34,071 |
| 8–15 | 373,561 | 30,059 |
| 16–31 | 363,530 | 29,364 |
| 32–63 | 428,535 | 34,355 |
| 64–127 | 410,624 | 33,039 |
| 128–255 | 203,997 | 16,162 |
| 256–511 | 28,098 | 2,222 |

Every revisited room in both recordings matches its earlier tiles exactly.

## Format

A recording directory holds three files.

| File | Contents |
|---|---|
| `steps.npz` | the four arrays below, one row per step |
| `meta.json` | format version 3, the settings that made the recording, and the count of rooms visited |
| `steps.gif` | the first steps rendered with the sprites, to check the navigation by eye |

| Array | Shape and type | Meaning |
|---|---|---|
| `tiles` | steps × 16 × 16, uint8 | the room without the character |
| `actions` | steps, uint8 | 0 up, 1 down, 2 left, 3 right |
| `episode` | steps, int64 | the world the step belongs to |
| `room` | steps, int64 | the room's index in that world's grid |

Training validates every array shape, dtype, token range, episode id, room id, and metadata
count, and records the SHA-256 of `steps.npz` in every checkpoint and summary. Evaluation must
use the recording that the checkpoint names.

## The game

A world is an eight by eight grid of rooms. Each room is 16 by 16 tiles and walled on every
side, with one door on each side that has a neighbouring room. The generator scatters patches of
grass, rock, tree, and water inside the walls and carves a corridor from every door to the middle
of the room, so every door can reach every other door.

Every episode generates a new world. A room's id is its episode's id times 256 plus its index in
the grid, and an episode's id carries the seed that made it. Two recordings made with different
seeds therefore share no room.

The character stands in the centre of the current room. One action moves it through a door to
the neighbouring room, where it stands in the centre again. The walk never steps straight back
when another door is available.

The sprites come from Drummyfish's public-domain
[1bit Graphics Collection](https://opengameart.org/content/1bit-graphics-collection).

## Reference points

Measure the reference points on the evaluation recording.

```sh
just baselines --recording data/roomnav_eval --out results/baselines/summary.json
```

Every guess is scored on the 196 cells inside the wall border, the same cells the probe scores.

| Guess for the room on a revisit | Accuracy |
|---|---:|
| The most common tile at each position | 0.42 |
| The previous step's frame | 0.31 |
| The frame from the last visit to the room | 1.00 |

Copying the last visit is exact, because a generated room never changes within its episode.
