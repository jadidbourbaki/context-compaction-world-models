"""Compact a transcript into a short walk over every room it visited, planned from actions alone."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from shared_cache_world_model.world import MOVES

# The action that moves from one position to a neighbouring one, keyed by the row and column step.
ACTION_BY_MOVE = {move: action for action, move in enumerate(MOVES)}


@dataclass(frozen=True)
class Walk:
    """A walk over every room a transcript visited, ending at the room it ends in."""

    # source_steps: (length,) int64. The transcript step whose frame stands for each room of the
    # walk, the latest step at that room, so a closed-loop harness reuses its newest prediction.
    source_steps: np.ndarray
    # actions: (length,) int64. The move from each room to the next. The last entry is the
    # transcript's own last action, the one about to be taken.
    actions: np.ndarray


@dataclass(frozen=True)
class Transcript:
    """Frames and actions in the shapes a model reads, one step per row."""

    # frames: (steps, 16, 16) uint8 room tiles without the character.
    frames: torch.Tensor
    # actions: (steps,) int64 indices into ACTIONS.
    actions: torch.Tensor


def positions_from_actions(actions: np.ndarray) -> np.ndarray:
    """The relative room position at every step, dead-reckoned from the actions before it."""
    if actions.ndim != 1:
        raise ValueError(f"actions must be one dimensional, got shape {actions.shape}")
    # moves: (steps, 2) int64, the row and column step each action takes.
    moves = np.asarray(MOVES, dtype=np.int64)[actions.astype(np.int64)]
    # positions: (steps, 2) int64. Step 0 sits at the origin and step t at the sum of the moves
    # before it, so two steps at one position are the same room.
    positions = np.zeros((actions.shape[0], 2), dtype=np.int64)
    positions[1:] = np.cumsum(moves[:-1], axis=0)
    return positions


def plan_walk(actions: np.ndarray) -> Walk:
    """Plan a short walk through every visited room that ends at the current one."""
    # The planner reads only the actions. Dead reckoning gives it the visited positions and the
    # current one without any frame.
    positions = positions_from_actions(actions)
    if positions.shape[0] == 0:
        raise ValueError("a transcript with no steps has no walk")
    current = (int(positions[-1, 0]), int(positions[-1, 1]))
    # latest_step: position -> the last transcript step that stood at it.
    latest_step: dict[tuple[int, int], int] = {}
    for step in range(positions.shape[0]):
        latest_step[(int(positions[step, 0]), int(positions[step, 1]))] = step
    visited = list(latest_step)
    # The visited positions form a connected region of the grid, because the character walked
    # between them. The region graph joins each position to its visited neighbours.
    region = nx.Graph()
    region.add_nodes_from(visited)
    for position in visited:
        for move in MOVES:
            neighbour = (position[0] + move[0], position[1] + move[1])
            if neighbour in latest_step:
                region.add_edge(position, neighbour)
    if len(visited) <= 2:
        # One or two rooms need no planning. The walk ends at the current room.
        route = [position for position in visited if position != current] + [current]
    else:
        # A traveling salesman approximation gives a near-shortest path through every position,
        # with consecutive positions adjacent in the region.
        route = nx.approximation.traveling_salesman_problem(region, cycle=False)
        # Close the walk at the current position from whichever end of the path is nearer.
        from_last = nx.shortest_path(region, route[-1], current)
        from_first = nx.shortest_path(region, route[0], current)
        if len(from_first) < len(from_last):
            reversed_route = list(reversed(route))
            route = reversed_route + from_first[1:]
        else:
            route = route + from_last[1:]
    # source_steps: (length,) int64, the newest transcript step at each position of the route.
    source_steps = np.array([latest_step[position] for position in route], dtype=np.int64)
    # walk_actions: (length,) int64, the move between consecutive positions of the route.
    walk_actions = np.zeros(len(route), dtype=np.int64)
    for index in range(len(route) - 1):
        move = (route[index + 1][0] - route[index][0], route[index + 1][1] - route[index][1])
        walk_actions[index] = ACTION_BY_MOVE[move]
    walk_actions[-1] = int(actions[-1])
    return Walk(source_steps=source_steps, actions=walk_actions)


def compact(actions: torch.Tensor, tiles: torch.Tensor) -> Transcript:
    """Compact a transcript of actions and room tiles into the frames and actions of its walk."""
    if tiles.shape[0] != actions.shape[0]:
        raise ValueError(f"{tiles.shape[0]} frames do not match {actions.shape[0]} actions")
    walk = plan_walk(actions.cpu().numpy())
    # frames: (length, 16, 16) uint8, each room's newest frame gathered from the transcript.
    frames = tiles[torch.from_numpy(walk.source_steps)]
    return Transcript(frames=frames, actions=torch.from_numpy(walk.actions))


def compact_when_full(actions: torch.Tensor, tiles: torch.Tensor, history: int) -> Transcript:
    """The context of a harness that appends raw steps and compacts only when it holds history."""
    if tiles.shape[0] != actions.shape[0]:
        raise ValueError(f"{tiles.shape[0]} frames do not match {actions.shape[0]} actions")
    # With walks of about 80 steps, a 128-step window compacts about once every 49 steps.
    context_steps = 0
    last_compaction: int | None = None
    for step in range(int(actions.shape[0])):
        if context_steps == history:
            walk = plan_walk(actions[:step].cpu().numpy())
            context_steps = len(walk.source_steps)
            if context_steps >= history:
                raise ValueError(f"a walk of {context_steps} steps leaves no room in {history}")
            last_compaction = step
        context_steps += 1
    if last_compaction is None:
        return Transcript(frames=tiles, actions=actions)
    kept = compact(actions[:last_compaction], tiles[:last_compaction])
    frames = torch.cat([kept.frames, tiles[last_compaction:]], dim=0)
    kept_actions = torch.cat([kept.actions, actions[last_compaction:]], dim=0)
    return Transcript(frames=frames, actions=kept_actions)


def newest_steps(actions: np.ndarray) -> np.ndarray:
    """The last transcript step at each visited room, in the order of those steps."""
    positions = positions_from_actions(actions)
    if positions.shape[0] == 0:
        raise ValueError("a transcript with no steps has no newest steps")
    latest_step: dict[tuple[int, int], int] = {}
    for step in range(positions.shape[0]):
        latest_step[(int(positions[step, 0]), int(positions[step, 1]))] = step
    # steps: (rooms,) int64. The final step is the transcript's last, so the current room ends it.
    steps = np.array(sorted(latest_step.values()), dtype=np.int64)
    return steps


def deduplicate(actions: torch.Tensor, tiles: torch.Tensor) -> Transcript:
    """Keep each room's newest step with its recorded action, with no walk between the rooms."""
    if tiles.shape[0] != actions.shape[0]:
        raise ValueError(f"{tiles.shape[0]} frames do not match {actions.shape[0]} actions")
    # source_steps: (rooms,) int64. Consecutive rooms need not be neighbours, so a recorded action
    # can point away from the room that follows it.
    source_steps = torch.from_numpy(newest_steps(actions.cpu().numpy()))
    return Transcript(frames=tiles[source_steps], actions=actions[source_steps])
