from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx
import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from shared_cache_world_model.agent import (
    Agent,
    ContextMode,
    StepRecord,
    donor_walk,
    plan_loop,
    reverse_action,
    summarize,
)
from shared_cache_world_model.data import TILES_PER_FRAME, model_frames
from shared_cache_world_model.model import ModelSettings
from shared_cache_world_model.world import (
    TILE_VOCAB,
    WATER,
    RoomWalker,
    WorldSettings,
    generate_world,
    move_room,
    valid_room_actions,
)


class Sentinel:
    """A stand-in model that predicts water in every cell and records what it was shown."""

    def __init__(self) -> None:
        self.settings = ModelSettings(kind="per_layer", d_model=32, n_heads=4, max_history=128)
        self.contexts: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def __call__(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.contexts.append(frames[0].clone())
        self.actions.append(actions[0].clone())
        # logits: (1, steps, 8, 256), highest for water at every cell of every step.
        logits = torch.zeros(frames.shape[0], frames.shape[1], TILE_VOCAB, TILES_PER_FRAME)
        logits[:, :, WATER] = 1.0
        return logits


@dataclass
class Setup:
    """An agent on a small generated world, with the model and walker that drive it."""

    world_settings: WorldSettings
    agent: Agent
    model: Sentinel
    walker: RoomWalker

    def walk(self, steps: int) -> list[StepRecord]:
        """Take ``steps`` random valid actions and return their records."""
        return [
            self.agent.step(self.walker.choose(self.world_settings, self.agent.room))
            for _ in range(steps)
        ]


def small_setup(seed: int, context: ContextMode = "compacted", dreaming: bool = True) -> Setup:
    # A three by three grid forces many revisits within a short walk.
    world_settings = WorldSettings(out=Path("unused"), rooms_across=3, rooms_down=3, seed=seed)
    rng = np.random.default_rng(seed)
    world = generate_world(rng, world_settings)
    start_room = int(rng.integers(9))
    model = Sentinel()
    agent = Agent(
        model,  # type: ignore[arg-type]
        world,
        world_settings,
        start_room,
        context,
        128,
        torch.device("cpu"),
        dreaming,
    )
    return Setup(world_settings=world_settings, agent=agent, model=model, walker=RoomWalker(rng))


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    context=st.sampled_from(["raw", "compacted"]),
    dreaming=st.booleans(),
)
@hypothesis_settings(max_examples=20, deadline=None)
def test_first_visits_are_observed_and_only_dreaming_enters_predictions(
    seed: int, context: str, dreaming: bool
) -> None:
    setup = small_setup(seed, context, dreaming)  # type: ignore[arg-type]
    water = torch.full_like(setup.agent.frames[0], WATER)
    records = setup.walk(30)
    revisits = [record for record in records if record.gap is not None]
    assert revisits
    # The model runs once per revisit and never on a first visit.
    assert len(setup.model.contexts) == len(revisits)
    for index, record in enumerate(records):
        entered = setup.agent.frames[index + 1]
        truth = setup.agent.true_room(record.room)
        if record.gap is None:
            assert torch.equal(entered, truth)
            assert record.remembered is None
        elif dreaming:
            # A dreamed revisit enters the model's water room into the transcript, however wrong.
            assert torch.equal(entered, water)
            assert record.remembered is not None
        else:
            # An observed revisit enters the true room and still scores the model's water room.
            assert torch.equal(entered, truth)
            assert record.remembered is not None
            assert record.remembered.accuracy < 1.0


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@hypothesis_settings(max_examples=10, deadline=None)
def test_a_revisit_is_scored_against_the_frame_the_agent_held(seed: int) -> None:
    setup = small_setup(seed)
    records = setup.walk(30)
    remembered_rooms: set[int] = set()
    for record in records:
        if record.remembered is None:
            continue
        # The first revisit compares the water room with the observed room, so self-consistency
        # equals accuracy. Every later revisit compares it with the model's own water room.
        if record.room in remembered_rooms:
            assert record.remembered.self_consistency == pytest.approx(1.0)
        else:
            assert record.remembered.self_consistency == pytest.approx(record.remembered.accuracy)
        assert record.remembered.accuracy < 1.0
        remembered_rooms.add(record.room)
    summary = summarize(records, (1, 8, 16, 32))
    revisit_count = sum(1 for record in records if record.gap is not None)
    assert sum(bin_summary.n for bin_summary in summary.values()) == revisit_count


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@hypothesis_settings(max_examples=10, deadline=None)
def test_a_compacted_context_ends_at_the_current_room_with_the_chosen_action(seed: int) -> None:
    setup = small_setup(seed)
    for _ in range(30):
        room_before = setup.agent.room
        action = setup.walker.choose(setup.world_settings, room_before)
        held_frame = model_frames(setup.agent.room_frames[room_before])
        calls_before = len(setup.model.contexts)
        # The play page's context meter reports this count before the step is taken.
        expected_steps = setup.agent.next_context_steps()
        setup.agent.step(action)
        if len(setup.model.contexts) == calls_before:
            continue
        assert setup.model.contexts[-1].shape[0] == expected_steps
        assert int(setup.model.actions[-1][-1]) == action
        assert torch.equal(setup.model.contexts[-1][-1], held_frame)
        # Nine rooms fit in a walk of at most eighteen steps.
        assert setup.model.contexts[-1].shape[0] <= 18


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), preserved=st.integers(1, 6))
@hypothesis_settings(max_examples=10, deadline=None)
def test_the_control_context_keeps_only_the_preserved_steps_of_the_session(
    seed: int, preserved: int
) -> None:
    # A control that kept a distant step would leak the target room and shrink the gain, and
    # one that dropped a preserved step would inflate it.
    setup = small_setup(seed, context="raw")
    donor = donor_walk(setup.world_settings, np.random.default_rng(seed + 1), 128)
    setup.agent.donor = donor
    setup.agent.preserved_steps = preserved
    records = setup.walk(40)
    revisits = [record for record in records if record.gap is not None]
    # The model reads the session context and then its control at every revisit.
    assert len(setup.model.contexts) == 2 * len(revisits)
    for index in range(len(revisits)):
        context = setup.model.contexts[2 * index]
        control = setup.model.contexts[2 * index + 1]
        steps = context.shape[0]
        kept = min(preserved, steps)
        assert control.shape[0] == steps
        assert torch.equal(control[steps - kept :], context[steps - kept :])
        assert torch.equal(control[: steps - kept], model_frames(donor.frames[: steps - kept]))
        assert revisits[index].remembered is not None
        assert revisits[index].remembered.shuffled_accuracy is not None


@given(
    room=st.integers(min_value=0, max_value=63),
    loop_steps=st.sampled_from([4, 6, 8, 16, 32, 64, 128]),
    previous_action=st.one_of(st.none(), st.integers(min_value=0, max_value=3)),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@hypothesis_settings(max_examples=300, deadline=None)
def test_a_loop_walks_grid_edges_without_reversing_and_first_returns_at_its_end(
    room: int, loop_steps: int, previous_action: int | None, seed: int
) -> None:
    world_settings = WorldSettings(out=Path("unused"), rooms_across=8, rooms_down=8, seed=0)
    rng = np.random.default_rng(seed)
    actions = plan_loop(world_settings, room, loop_steps, previous_action, rng)
    assert len(actions) == loop_steps
    grid = networkx.grid_2d_graph(8, 8)
    rooms = [room]
    for action in actions:
        rooms.append(move_room(world_settings, rooms[-1], action))
    for index in range(loop_steps):
        here = divmod(rooms[index], 8)
        there = divmod(rooms[index + 1], 8)
        assert grid.has_edge(here, there)
    # Training never steps straight back, so neither does the loop, from its first move on.
    moves = [previous_action, *actions] if previous_action is not None else actions
    for index in range(len(moves) - 1):
        assert moves[index + 1] != reverse_action(moves[index])
    # The start room appears only at the two ends, so its revisit comes at gap loop_steps.
    assert rooms[-1] == room
    assert room not in rooms[1:-1]


def test_a_loop_that_no_walk_can_close_raises() -> None:
    # The only walk on a 2 by 2 grid that never steps back circles the 4 rooms, so it returns
    # to its start after 4 steps and a 6-step loop cannot exist.
    world_settings = WorldSettings(out=Path("unused"), rooms_across=2, rooms_down=2, seed=0)
    with pytest.raises(ValueError, match="no 6-step loop found around room 0"):
        plan_loop(world_settings, 0, 6, None, np.random.default_rng(42))


def test_the_agent_refuses_a_move_through_a_wall() -> None:
    setup = small_setup(0)
    room = setup.agent.room
    blocked = [
        action
        for action in range(4)
        if action not in valid_room_actions(setup.world_settings, room)
    ]
    assert blocked
    with pytest.raises(ValueError, match="invalid from room"):
        setup.agent.step(blocked[0])
