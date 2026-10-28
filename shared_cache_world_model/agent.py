"""Run a trained world model in closed loop, the way an agent plays the game through it."""

# The agent plays the role of the program that calls the model. The agent knows the rules of
# the game and the index of every room, because moving between rooms is the game's job. On the
# first visit to a room the agent observes it, so the true room enters the transcript. On a
# revisit the agent observes nothing and asks the model for the room from memory, so the model's
# prediction enters the transcript and every later revisit builds on it. The agent dreams its
# revisits in this mode. With dreaming off, the agent observes every room, so the model still
# predicts each revisit and is scored on it while the true room enters the transcript. The
# generated world is read only for observations and to score the remembered rooms. At every
# revisit the agent also predicts from a control context, in which the steps before the last few
# come from a walk through another world, so the score separates retrieval from local guesses
# exactly as the probe's shuffled run does.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
import tyro
from pydantic import BaseModel, Field, FilePath, model_validator

from shared_cache_world_model.compaction import (
    ACTION_BY_MOVE,
    Transcript,
    compact,
    compact_when_full,
    plan_walk,
)
from shared_cache_world_model.data import model_frames
from shared_cache_world_model.model import ModelSettings, WorldModel
from shared_cache_world_model.probe import bin_label
from shared_cache_world_model.revisits import validate_gap_bins
from shared_cache_world_model.world import (
    MOVES,
    ROOM_COLS,
    ROOM_ROWS,
    SCORED_CELLS,
    RoomWalker,
    World,
    WorldSettings,
    generate_world,
    move_room,
    valid_room_actions,
)

ContextMode = Literal["raw", "compacted", "compacted_when_full"]
# walk: the recording walker, which never steps straight back. loop: walks that first return to
# their start room after a fixed number of steps, with walker steps between them.
Policy = Literal["walk", "loop"]
# A random loop can dead-end against the grid border, so the planner draws up to 100 walks.
LOOP_ATTEMPTS = 100
logger = logging.getLogger(__name__)


class FrameModel(Protocol):
    """What the agent needs from a model: its settings and next-frame logits."""

    settings: ModelSettings

    def __call__(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor: ...


class AgentSettings(BaseModel):
    checkpoint: FilePath = Field(description="weights.pt or checkpoint.pt written by train")
    # raw: the last history steps. compacted: the walk on every call. compacted_when_full: raw
    # steps until the window fills, then the walk.
    context: ContextMode = Field(description="raw, compacted, or compacted_when_full")
    seed: int = Field(default=43, description="world seed, which must differ from training")
    dreaming: bool = Field(default=True, description="revisits enter the transcript as predicted")
    episodes: int = Field(default=10, ge=1)
    steps: int = Field(default=512, ge=1, description="room transitions per episode")
    policy: Policy = Field(default="walk")
    loop_steps: int = Field(default=8, ge=4, description="steps per loop under the loop policy")
    # Revisits are grouped into the same gap bins the probe uses by default.
    bins: tuple[int, ...] = Field(default=(1, 8, 16, 32, 64, 128, 256, 512))
    device: str | None = Field(default=None)
    out: Path = Field(description="JSON report path")
    # A 512-step episode records 120 KB of steps, so a batch keeps only the summaries.
    save_records: bool = Field(default=True, description="write every step into the report")

    @model_validator(mode="after")
    def validate_bins_and_loop(self) -> AgentSettings:
        validate_gap_bins(self.bins)
        if self.loop_steps % 2 != 0:
            # The room grid is bipartite, so only an even number of moves returns to a room.
            raise ValueError(f"loop_steps {self.loop_steps} must be even to return to its room")
        if self.policy == "loop" and self.steps < self.loop_steps:
            raise ValueError(f"steps {self.steps} cannot hold one {self.loop_steps}-step loop")
        return self


class Remembered(BaseModel):
    """How a room the model supplied from memory scored."""

    # Agreement with the true room on the scored cells.
    accuracy: float
    # Agreement with the frame the agent holds for this room from the last visit, which is the
    # observation after one visit and the model's own earlier prediction after two or more.
    self_consistency: float
    # Mean agreement with the frames the agent holds for every other room, the control that a
    # prediction resembling any room would pass.
    other_room_consistency: float
    # Agreement with the true room when every step before the model's local receptive field
    # comes from another world, None when the agent runs without a control.
    shuffled_accuracy: float | None
    context_steps: int


class StepRecord(BaseModel):
    """One room transition, observed on a first visit and remembered on a revisit."""

    step: int
    room: int
    # Steps since the character last stood in this room, None on a first visit.
    gap: int | None
    # The model's room and its scores on a revisit, None on a first visit, which is observed.
    remembered: Remembered | None


class BinSummary(BaseModel):
    accuracy: float
    self_consistency: float
    other_room_consistency: float
    # Self-consistency minus other-room consistency: how much of the agreement is specific to
    # the room, the closed-loop counterpart of the probe's causal retrieval gain.
    consistency_gain: float
    # Accuracy minus the accuracy of the control context, the probe's context retrieval gain.
    shuffled_accuracy: float | None
    retrieval_gain: float | None
    n: int


class AgentReport(BaseModel):
    config: dict[str, object]
    # Revisits grouped by the steps since the room was last entered.
    by_gap: dict[str, BinSummary]
    # Under the loop policy, the revisits at exactly the loop length, one per closed loop.
    loop_gap: BinSummary | None
    # Every step of every episode, empty when the run keeps only its summaries.
    records: list[StepRecord]


def agreement(first: torch.Tensor, second: torch.Tensor) -> float:
    """The fraction of scored cells on which two (16, 16) room frames hold the same tile."""
    scored = torch.from_numpy(SCORED_CELLS)
    return float((first[scored] == second[scored]).float().mean())


class Agent:
    """Hold a transcript of predicted rooms and step it forward one action at a time."""

    def __init__(
        self,
        model: FrameModel,
        world: World,
        world_settings: WorldSettings,
        start_room: int,
        context: ContextMode,
        history: int,
        device: torch.device,
        dreaming: bool,
        donor: Transcript | None = None,
        preserved_steps: int = 0,
    ) -> None:
        self.model = model
        self.world = world
        self.world_settings = world_settings
        self.context = context
        self.history = history
        self.device = device
        # The play page flips dreaming and the context mode between steps, so both stay public.
        self.dreaming = dreaming
        # donor: steps of a walk through another world that replace the distant part of the
        # control context. preserved_steps: the steps at the end of the context that stay.
        self.donor = donor
        self.preserved_steps = preserved_steps
        self.room = start_room
        self.step_count = 0
        # frames: one (16, 16) uint8 room per step, observed on a first visit and predicted on a
        # revisit while dreaming.
        self.frames: list[torch.Tensor] = [self.true_room(start_room)]
        # view: (16, 16) uint8, the room the model saw or remembered on the latest step. A revisit
        # shows the prediction whether or not the prediction entered the transcript.
        self.view = self.frames[0]
        # actions: the action taken from each step, one fewer than frames between steps.
        self.actions: list[int] = []
        # The newest frame and step the agent holds for each room it has entered.
        self.room_frames: dict[int, torch.Tensor] = {start_room: self.frames[0]}
        self.room_steps: dict[int, int] = {start_room: 0}

    def true_room(self, room: int) -> torch.Tensor:
        """The generated room, which the agent reads to observe a first visit and to score."""
        row = room // self.world_settings.rooms_across
        col = room % self.world_settings.rooms_across
        return torch.from_numpy(self.world.tiles[row, col].copy())

    def valid_actions(self) -> list[int]:
        """The moves the game allows from the current room."""
        return valid_room_actions(self.world_settings, self.room)

    def model_context(self) -> Transcript:
        """The frames and actions the agent sends the model, ending with the current step."""
        # torch.stack joins the per-step rooms into one tensor. frames: (steps, 16, 16) uint8.
        frames = torch.stack(self.frames)
        # actions: (steps,) int64, one per frame, the last being the action just taken.
        actions = torch.tensor(self.actions, dtype=torch.int64)
        if self.context == "raw":
            start = max(0, frames.shape[0] - self.history)
            return Transcript(frames=frames[start:], actions=actions[start:])
        if self.context == "compacted_when_full":
            return compact_when_full(actions, frames, self.history)
        return compact(actions, frames)

    def next_context_steps(self) -> int:
        """The steps the model would read if the next room were a revisit."""
        steps = len(self.frames)
        if self.context == "raw":
            return min(steps, self.history)
        if self.context == "compacted_when_full":
            # actions: (steps,) int64 with a placeholder for the action not yet taken.
            actions = torch.tensor([*self.actions, 0], dtype=torch.int64)
            frames = torch.stack(self.frames)
            return int(compact_when_full(actions, frames, self.history).frames.shape[0])
        # The walk's rooms depend only on the actions before the last step, so a placeholder
        # for the next action gives the same length as the action the agent later takes.
        actions = np.array([*self.actions, 0], dtype=np.int64)
        return len(plan_walk(actions).source_steps)

    def control_context(self, context: Transcript) -> Transcript:
        """The context with every step before the last preserved ones taken from the donor."""
        if self.donor is None:
            raise ValueError("the agent has no donor walk for a control context")
        steps = int(context.frames.shape[0])
        shuffled_steps = steps - self.preserved_steps
        if shuffled_steps <= 0:
            return context
        if shuffled_steps > int(self.donor.frames.shape[0]):
            raise ValueError(
                f"the donor walk holds {self.donor.frames.shape[0]} steps, {shuffled_steps} needed"
            )
        # The donor's first steps replace the context's first steps, so the target room and the
        # route into it survive while the distant evidence comes from another world.
        frames = torch.cat([self.donor.frames[:shuffled_steps], context.frames[shuffled_steps:]])
        actions = torch.cat([self.donor.actions[:shuffled_steps], context.actions[shuffled_steps:]])
        return Transcript(frames=frames, actions=actions)

    def predict(self, context: Transcript) -> torch.Tensor:
        """The model's next room after the context's last step: (16, 16) uint8."""
        steps = int(context.frames.shape[0])
        if steps > self.model.settings.max_history:
            raise ValueError(
                f"a context of {steps} steps exceeds the model's max_history "
                f"{self.model.settings.max_history}"
            )
        # frames: (1, steps, 16, 16) int64 with the character drawn in. actions: (1, steps).
        frames = model_frames(context.frames)[None].long().to(self.device)
        actions = context.actions[None].to(self.device)
        # torch.no_grad skips recording the graph, because the agent never trains the model.
        with torch.no_grad():
            # logits: (1, steps, 8, 256). The last step's outputs predict the room entered next.
            logits = self.model(frames, actions)
        # argmax over the 8 tile types picks each cell's most likely tile. predicted: (256,) int64.
        predicted = logits[0, -1].argmax(dim=0)
        return predicted.reshape(ROOM_ROWS, ROOM_COLS).to(torch.uint8).cpu()

    def step(self, action: int) -> StepRecord:
        """Take one action and enter the next room, observed if new and remembered if not."""
        if action not in self.valid_actions():
            raise ValueError(f"action {action} is invalid from room {self.room}")
        self.actions.append(action)
        next_room = move_room(self.world_settings, self.room, action)
        held = self.room_frames.get(next_room)
        # truth: (16, 16) uint8, the generated room.
        truth = self.true_room(next_room)
        if held is None:
            # A first visit: the agent observes the room.
            entered = truth
            self.view = truth
            remembered = None
            gap = None
        else:
            # A revisit: the model supplies the room from the transcript the agent sends it.
            context = self.model_context()
            predicted = self.predict(context)
            # Dreaming puts the prediction in the transcript. Otherwise the agent observes.
            entered = predicted if self.dreaming else truth
            self.view = predicted
            other_frames = [frame for room, frame in self.room_frames.items() if room != next_room]
            shuffled_accuracy = None
            if self.donor is not None:
                control = self.predict(self.control_context(context))
                shuffled_accuracy = agreement(control, truth)
            remembered = Remembered(
                accuracy=agreement(predicted, truth),
                self_consistency=agreement(predicted, held),
                other_room_consistency=float(
                    np.mean([agreement(predicted, frame) for frame in other_frames])
                ),
                shuffled_accuracy=shuffled_accuracy,
                context_steps=int(context.frames.shape[0]),
            )
            gap = self.step_count + 1 - self.room_steps[next_room]
        self.step_count += 1
        self.frames.append(entered)
        self.room = next_room
        self.room_frames[next_room] = entered
        self.room_steps[next_room] = self.step_count
        return StepRecord(step=self.step_count, room=next_room, gap=gap, remembered=remembered)


def load_model(checkpoint: Path, device: torch.device) -> tuple[WorldModel, dict[str, object]]:
    """Rebuild a trained model and return it with the saved training state."""
    # A checkpoint holds the config and recording identity as plain dicts beside the weights, so
    # the loader needs full unpickling. map_location puts the tensors on the chosen device.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = WorldModel(ModelSettings.model_validate(config["model"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def reverse_action(action: int) -> int:
    """The action that undoes ``action``."""
    move = MOVES[action]
    return ACTION_BY_MOVE[(-move[0], -move[1])]


def room_distance(world_settings: WorldSettings, first: int, second: int) -> int:
    """The fewest moves between two rooms, their Manhattan distance on the room grid."""
    row_steps = abs(first // world_settings.rooms_across - second // world_settings.rooms_across)
    column_steps = abs(first % world_settings.rooms_across - second % world_settings.rooms_across)
    return row_steps + column_steps


def plan_loop(
    world_settings: WorldSettings,
    room: int,
    loop_steps: int,
    previous_action: int | None,
    rng: np.random.Generator,
) -> list[int]:
    """A random walk of ``loop_steps`` moves that returns to ``room`` on its last move only."""
    if loop_steps < 4 or loop_steps % 2 != 0:
        raise ValueError(f"a loop of {loop_steps} steps must be even and at least 4")
    for _ in range(LOOP_ATTEMPTS):
        actions: list[int] = []
        current = room
        last_action = previous_action
        while len(actions) < loop_steps:
            remaining = loop_steps - len(actions)
            distance = room_distance(world_settings, current, room)
            # Slack is the number of spare steps beyond the shortest way home. The room grid
            # is bipartite, so slack is always even.
            slack = remaining - distance
            choices = []
            for action in valid_room_actions(world_settings, current):
                # Stepping straight back never occurs in training.
                if last_action is not None and action == reverse_action(last_action):
                    continue
                next_room = move_room(world_settings, current, action)
                if next_room == room and remaining > 1:
                    continue
                closer = room_distance(world_settings, next_room, room) < distance
                # A move away from home spends 2 steps of slack. With 2 left, the move must
                # land off the start room's row and column, so the room it lands in has a way
                # home that does not step straight back.
                off_axis = (
                    next_room // world_settings.rooms_across != room // world_settings.rooms_across
                    and next_room % world_settings.rooms_across
                    != room % world_settings.rooms_across
                )
                if closer or slack >= 4 or (slack == 2 and off_axis):
                    choices.append(action)
            if not choices:
                break
            action = choices[int(rng.integers(len(choices)))]
            actions.append(action)
            current = move_room(world_settings, current, action)
            last_action = action
        if len(actions) == loop_steps:
            return actions
    raise ValueError(f"no {loop_steps}-step loop found around room {room}")


def donor_walk(world_settings: WorldSettings, rng: np.random.Generator, steps: int) -> Transcript:
    """A walk of ``steps`` observed rooms through a newly generated world."""
    world = generate_world(rng, world_settings)
    room_count = world_settings.rooms_across * world_settings.rooms_down
    room = int(rng.integers(room_count))
    walker = RoomWalker(rng)
    frames = []
    actions = []
    for _ in range(steps):
        row = room // world_settings.rooms_across
        col = room % world_settings.rooms_across
        frames.append(torch.from_numpy(world.tiles[row, col].copy()))
        action = walker.choose(world_settings, room)
        actions.append(action)
        room = move_room(world_settings, room, action)
    # frames: (steps, 16, 16) uint8. actions: (steps,) int64.
    return Transcript(frames=torch.stack(frames), actions=torch.tensor(actions, dtype=torch.int64))


def evaluation_world_settings(state: dict[str, object], seed: int) -> WorldSettings:
    """The training world distribution under a seed the model never trained on."""
    recording = state["recording"]
    if not isinstance(recording, dict):
        raise ValueError("checkpoint lacks its recording identity")
    world_settings = WorldSettings.model_validate(recording["settings"])
    if world_settings.seed == seed:
        raise ValueError(f"world seed {seed} is the training seed")
    return world_settings.model_copy(update={"seed": seed})


def training_history(state: dict[str, object]) -> int:
    """The window length, in steps, the model trained on."""
    config = state["config"]
    if not isinstance(config, dict):
        raise ValueError("checkpoint lacks its training config")
    return int(config["history"])


def summarize(records: list[StepRecord], bins: tuple[int, ...]) -> dict[str, BinSummary]:
    """Mean accuracy and consistency of revisits by gap bin."""
    grouped: dict[str, list[tuple[float, float, float, float]]] = {}
    for record in records:
        if record.gap is None:
            continue
        remembered = record.remembered
        if remembered is None:
            raise ValueError(f"revisit at step {record.step} has no remembered room")
        # A run without a control scores its control as NaN, which the means carry through.
        shuffled = remembered.shuffled_accuracy
        scores = (
            remembered.accuracy,
            remembered.self_consistency,
            remembered.other_room_consistency,
            float("nan") if shuffled is None else shuffled,
        )
        grouped.setdefault(bin_label(record.gap, bins), []).append(scores)
    summaries: dict[str, BinSummary] = {}
    for label, members in grouped.items():
        # table: (revisits, 4) float64, accuracy, self-consistency, other-room consistency, and
        # control accuracy.
        table = np.asarray(members)
        means = table.mean(axis=0)
        has_control = not np.isnan(means[3])
        summaries[label] = BinSummary(
            accuracy=float(means[0]),
            self_consistency=float(means[1]),
            other_room_consistency=float(means[2]),
            consistency_gain=float(means[1] - means[2]),
            shuffled_accuracy=float(means[3]) if has_control else None,
            retrieval_gain=float(means[0] - means[3]) if has_control else None,
            n=len(members),
        )
    return summaries


def main(settings: AgentSettings) -> None:
    device = (
        torch.device(settings.device)
        if settings.device
        else torch.accelerator.current_accelerator()
    )
    if device is None:
        raise RuntimeError("no accelerator found. Pass --device cpu to run on the CPU")
    loaded = load_model(Path(settings.checkpoint), device)
    model = loaded[0]
    state = loaded[1]
    # The worlds come from the training distribution with a seed training never used, so every
    # room the agent enters is new to the model.
    world_settings = evaluation_world_settings(state, settings.seed)
    history = training_history(state)
    rng = np.random.default_rng(settings.seed)
    records: list[StepRecord] = []
    # The revisit that closes each loop, whose gap is the loop length.
    loop_closings: list[StepRecord] = []
    for episode in range(settings.episodes):
        world = generate_world(rng, world_settings)
        room_count = world_settings.rooms_across * world_settings.rooms_down
        start_room = int(rng.integers(room_count))
        # The donor comes from its own generator, so the worlds and walks of the agent stay the
        # same with or without the control.
        donor_rng = np.random.default_rng((settings.seed, episode))
        donor = donor_walk(world_settings, donor_rng, model.settings.max_history)
        agent = Agent(
            model,
            world,
            world_settings,
            start_room,
            settings.context,
            history,
            device,
            settings.dreaming,
            donor=donor,
            preserved_steps=model.local_receptive_steps(),
        )
        walker = RoomWalker(rng)
        # Under the loop policy, 8 walker steps after an 8-step loop carry the agent to new rooms
        # before the next loop begins.
        walker_steps = settings.loop_steps if settings.policy == "loop" else settings.steps
        while agent.step_count < settings.steps:
            if settings.policy == "loop":
                previous_action = agent.actions[-1] if agent.actions else None
                loop = plan_loop(
                    world_settings, agent.room, settings.loop_steps, previous_action, rng
                )
                for action in loop[: settings.steps - agent.step_count]:
                    # The walker reads the room the agent left so its next step does not go back.
                    walker.previous_room = agent.room
                    records.append(agent.step(action))
                if records[-1].gap == settings.loop_steps:
                    loop_closings.append(records[-1])
            for _ in range(min(walker_steps, settings.steps - agent.step_count)):
                records.append(agent.step(walker.choose(world_settings, agent.room)))
        logger.info("episode %d of %d", episode + 1, settings.episodes)
    loop_gap = None
    if settings.policy == "loop":
        # The bins 8 and 9 hold exactly the revisits at gap 8, labelled "8-8".
        loop_bins = (settings.loop_steps, settings.loop_steps + 1)
        loop_label = bin_label(settings.loop_steps, loop_bins)
        loop_gap = summarize(loop_closings, loop_bins)[loop_label]
    report = AgentReport(
        config=settings.model_dump(mode="json"),
        by_gap=summarize(records, settings.bins),
        loop_gap=loop_gap,
        records=records if settings.save_records else [],
    )
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    settings.out.write_text(report.model_dump_json(indent=2) + "\n")
    summary = report.model_dump(mode="json")["by_gap"]
    logger.info("by gap %s", json.dumps(summary))
    if loop_gap is not None:
        logger.info("at the loop gap %s", loop_gap.model_dump_json())
    logger.info("wrote %s", settings.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(AgentSettings))
