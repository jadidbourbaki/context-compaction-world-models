"""Play the generated game inside a trained world model in the browser."""

# The page shows a map of the rooms the agent has entered with the planned route, the agent's
# room, and the true room. The agent's room is the true room while observing a first visit and the
# model's room while predicting a revisit. A loop button plans a walk of 4 to 128 room
# transitions that first returns to the current room on its last step, and the random walk
# buttons take 8 or 16 recording-walker steps to reach new rooms. The model selector swaps the
# architecture that remembers, keeping the transcript. The dreaming box decides whether a
# predicted room enters the transcript, and the context selector chooses the raw window, the
# compacted walk on every step, or raw steps compacted once the window fills.

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import tyro
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from shared_cache_world_model.agent import (
    Agent,
    ContextMode,
    StepRecord,
    donor_walk,
    evaluation_world_settings,
    load_model,
    plan_loop,
    training_history,
)
from shared_cache_world_model.model import WorldModel
from shared_cache_world_model.world import (
    RoomWalker,
    add_character,
    generate_world,
    move_room,
    render_room,
    sprite_atlas,
)

WEIGHTS_REPO = "jadidbourbaki/context-compaction-world-models"
# The selector names each architecture as the paper does and maps it to its seed 42 run.
MODELS = {
    "Decoder-only": "roomnav_transition_v1_per_layer_h128_s42",
    "CED": "roomnav_transition_v1_ced_full_h128_s42",
    "CED-SWA": "roomnav_transition_v1_ced_h128_s42",
    "DD": "roomnav_transition_v1_shared_cache_h128_s42",
}
# The context selector names each agent harness context as the paper does.
CONTEXTS: dict[str, ContextMode] = {
    "Raw Window": "raw",
    "Context Compaction": "compacted",
    "Compact When Full": "compacted_when_full",
}
# At the end of a 128-step loop the start room is the oldest frame of the 128-step raw window.
LOOP_LENGTHS = (4, 8, 16, 32, 64, 128)
WALK_LENGTHS = (8, 16)
STEP_SECONDS = 0.25
# A rendered room is 16 tiles of 16 pixels inside a one-tile border, 288 pixels square.
PANEL = 288
# The page is as wide as three 288 pixel columns and the two 6 pixel gaps between them.
PAGE = 3 * PANEL + 2 * 6
# The map draws each tile as 2 by 2 pixels, 32 pixels per room, with 4 pixel grid lines.
MAP_CELL = 32
MAP_LINE = 4
MAP_PITCH = MAP_CELL + MAP_LINE
GRID = (70, 70, 70)
UNEXPLORED = (16, 16, 16)
RED = (230, 57, 70)
# The Windows 95 theme from the Gradio theme gallery, pinned to one published version.
THEME = "hmb/windows95@0.0.7"
# Each settings row centres its label beside its picker. Gradio grows a Markdown block like a
# picker whatever its scale, so the label is held to its text width. At 12 pixels
# "Context Compaction" clears the picker's arrow, and the open lists use the same size. Each
# readout is one line, and the context bar stretches between its label and its count. The theme
# fills the bar with 8 pixel blocks, so 15 of 128 steps cuts a block into a sliver. A solid fill
# avoids the sliver, and 2 pixels of padding show the theme's sunken border, which the browser
# otherwise paints over.
CSS = """
.settings { align-items: center; }
.model-label { flex: none !important; width: fit-content !important; }
.model-picker input, .model-picker .options { font-size: 12px !important; }
.readout { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.readout progress { flex: 1; appearance: none; padding: 2px; }
.readout progress::-webkit-progress-value { background: #000080 !important; }
.readout progress::-moz-progress-bar { background: #000080 !important; }
"""


class PlaySettings(BaseModel):
    seed: int = Field(default=43, description="first world seed, which must differ from training")
    device: str | None = Field(default=None)
    open_browser: bool = Field(default=True, description="open the page in a local browser")


class Session:
    """One visitor's world, agent, and walker."""

    def __init__(
        self, model: WorldModel, state: dict[str, object], seed: int, context: ContextMode
    ) -> None:
        self.seed = seed
        self.world_settings = evaluation_world_settings(state, seed)
        rng = np.random.default_rng(seed)
        world = generate_world(rng, self.world_settings)
        room_count = self.world_settings.rooms_across * self.world_settings.rooms_down
        start_room = int(rng.integers(room_count))
        history = training_history(state)
        device = next(model.parameters()).device
        # Every architecture keeps the same local receptive field, so a model swap keeps the
        # control. The donor has its own generator, so the world and walks match the agent batch.
        donor = donor_walk(self.world_settings, np.random.default_rng((seed, 0)), history)
        self.agent = Agent(
            model,
            world,
            self.world_settings,
            start_room,
            context,
            history,
            device,
            True,
            donor=donor,
            preserved_steps=model.local_receptive_steps(),
        )
        # The generator draws the walker's steps and the loop planner's walks.
        self.rng = rng
        self.walker = RoomWalker(rng)
        self.record: StepRecord | None = None
        # The latest revisit, whose accuracy stays on screen through the observed steps after it.
        self.last_revisit: StepRecord | None = None
        self.queued: list[int] = []


# The session, the map, the agent's heading, the agent's room, the true room, the latest
# revisit's accuracy with the context the next prediction reads.
Outputs = tuple[Session, np.ndarray, str, np.ndarray, np.ndarray, str]


@cache
def load_run(run: str, device: str) -> tuple[WorldModel, dict[str, object]]:
    """Download a run's weights from Hugging Face once and load them."""
    path = hf_hub_download(WEIGHTS_REPO, f"{run}/weights.pt")
    return load_model(Path(path), torch.device(device))


def tile_colours() -> np.ndarray:
    """(8, 3) uint8 map colour per tile type: floor and grass white, water and doors black."""
    brightness = sprite_atlas().mean(axis=(1, 2, 3))
    shade = np.select([brightness > 180, brightness < 100], [255, 0], 140).astype(np.uint8)
    return np.repeat(shade[:, None], 3, axis=1)


def render_map(session: Session) -> np.ndarray:
    """The room grid with entered rooms as thumbnails, the planned route, and the current room."""
    settings = session.world_settings
    colours = tile_colours()
    size = settings.rooms_across * MAP_PITCH + MAP_LINE
    # pixels: (292, 292, 3) uint8 for the 8 by 8 grid, with grid lines around every room.
    pixels = np.full((size, size, 3), GRID, dtype=np.uint8)
    room_count = settings.rooms_across * settings.rooms_down
    for room in range(room_count):
        top = MAP_LINE + room // settings.rooms_across * MAP_PITCH
        left = MAP_LINE + room % settings.rooms_across * MAP_PITCH
        cell = pixels[top : top + MAP_CELL, left : left + MAP_CELL]
        if room in session.agent.room_steps:
            tiles = session.agent.true_room(room).numpy()
            # Repeating along both axes doubles each tile into a 2 by 2 block.
            cell[:] = colours[tiles].repeat(2, axis=0).repeat(2, axis=1)
        else:
            cell[:] = UNEXPLORED
    rooms = [session.agent.room]
    for action in session.queued:
        rooms.append(move_room(settings, rooms[-1], action))
    # Every move is between neighbouring rooms, so each leg of the route is a straight bar
    # 4 pixels wide from one room centre to the next.
    centres = [
        (
            MAP_LINE + room // settings.rooms_across * MAP_PITCH + MAP_CELL // 2,
            MAP_LINE + room % settings.rooms_across * MAP_PITCH + MAP_CELL // 2,
        )
        for room in rooms
    ]
    for index in range(len(centres) - 1):
        start = centres[index]
        end = centres[index + 1]
        top = min(start[0], end[0]) - 2
        bottom = max(start[0], end[0]) + 2
        left = min(start[1], end[1]) - 2
        right = max(start[1], end[1]) + 2
        pixels[top:bottom, left:right] = RED
    current = centres[0]
    top = current[0] - MAP_CELL // 2 - MAP_LINE
    left = current[1] - MAP_CELL // 2 - MAP_LINE
    outline = pixels[top : top + MAP_PITCH + MAP_LINE, left : left + MAP_PITCH + MAP_LINE]
    outline[:MAP_LINE] = RED
    outline[-MAP_LINE:] = RED
    outline[:, :MAP_LINE] = RED
    outline[:, -MAP_LINE:] = RED
    # Trimming half a grid line from each edge makes the map 288 pixels, the size of a room.
    return pixels[MAP_LINE // 2 : -MAP_LINE // 2, MAP_LINE // 2 : -MAP_LINE // 2]


def agent_title(session: Session) -> str:
    """The agent panel's title with the agent's stage in brackets."""
    predicting = session.record is not None and session.record.remembered is not None
    stage = "Predicting" if predicting else "Observing"
    return f"### Agent [{stage}]"


def readout(session: Session) -> str:
    """The latest revisit's accuracy and gain, and a bar of the context the next step reads."""
    revisit = session.last_revisit
    accuracy = "N/A"
    gain = "N/A"
    if revisit is not None and revisit.remembered is not None:
        remembered = revisit.remembered
        accuracy = f"{100 * remembered.accuracy:.2f}%"
        if remembered.shuffled_accuracy is None:
            raise ValueError("the play page's agent runs without a control context")
        gain = f"{100 * (remembered.accuracy - remembered.shuffled_accuracy):+.2f}%"
    context_steps = session.agent.next_context_steps()
    history = session.agent.history
    # The theme draws the browser's own progress bar in its Windows 95 style.
    bar = f"<progress value='{context_steps}' max='{history}'></progress>"
    return (
        f"<div class='readout'><b>Accuracy:</b> {accuracy}</div>"
        f"<div class='readout'><b>Context Retrieval Gain:</b> {gain}</div>"
        f"<div class='readout'><b>Context:</b> {bar} {context_steps} of {history} steps</div>"
    )


def view(session: Session) -> Outputs:
    """The session and every output the page redraws after a change."""
    if session.record is not None and session.record.remembered is not None:
        session.last_revisit = session.record
    agent_tiles = add_character(session.agent.view.numpy())
    true_tiles = add_character(session.agent.true_room(session.agent.room).numpy())
    # agent_room and true_room: (288, 288, 3) uint8 sprite renders.
    agent_room = render_room(agent_tiles)
    true_room = render_room(true_tiles)
    return (
        session,
        render_map(session),
        agent_title(session),
        agent_room,
        true_room,
        readout(session),
    )


def room_image() -> gr.Image:
    """An image without the title badge and the download and share buttons."""
    # Each image is exactly one room in size, so no frame surrounds it.
    return gr.Image(
        show_label=False,
        interactive=False,
        buttons=[],
        container=False,
        width=PANEL,
        height=PANEL,
    )


def move_handler(
    move: Callable[[Session, int], Iterator[Outputs]], length: int
) -> Callable[[Session], Iterator[Outputs]]:
    """A click handler that plays one loop or walk of a fixed length."""

    # Gradio streams a handler's updates only when the handler itself is a generator function.
    def handler(session: Session) -> Iterator[Outputs]:
        yield from move(session, length)

    return handler


def build(settings: PlaySettings) -> gr.Blocks:
    """The page and its event handlers."""
    device = (
        torch.device(settings.device)
        if settings.device
        else torch.accelerator.current_accelerator()
    )
    if device is None:
        raise RuntimeError("no accelerator found. Pass --device cpu to run on the CPU")
    device_name = str(device)

    def start(model_name: str, context_name: str, seed: int) -> Outputs:
        loaded = load_run(MODELS[model_name], device_name)
        return view(Session(loaded[0], loaded[1], seed, CONTEXTS[context_name]))

    def new_world(session: Session, model_name: str, context_name: str) -> Outputs:
        return start(model_name, context_name, session.seed + 1)

    def play(session: Session, actions: list[int]) -> Iterator[Outputs]:
        session.queued = actions
        while session.queued:
            # The walker reads the room the agent left so a walk after a loop does not step back.
            session.walker.previous_room = session.agent.room
            session.record = session.agent.step(session.queued.pop(0))
            yield view(session)
            time.sleep(STEP_SECONDS)

    def loop(session: Session, loop_steps: int) -> Iterator[Outputs]:
        previous_action = session.agent.actions[-1] if session.agent.actions else None
        actions = plan_loop(
            session.world_settings, session.agent.room, loop_steps, previous_action, session.rng
        )
        yield from play(session, actions)

    def walk(session: Session, walk_steps: int) -> Iterator[Outputs]:
        actions = []
        room = session.agent.room
        for _ in range(walk_steps):
            action = session.walker.choose(session.world_settings, room)
            actions.append(action)
            room = move_room(session.world_settings, room, action)
        yield from play(session, actions)

    def select_model(session: Session, model_name: str) -> Outputs:
        # The transcript belongs to the agent, so a new model continues from the same history.
        session.agent.model = load_run(MODELS[model_name], device_name)[0]
        return view(session)

    def set_dreaming(session: Session, dreaming: bool) -> None:
        session.agent.dreaming = dreaming

    def set_context(session: Session, context_name: str) -> str:
        session.agent.context = CONTEXTS[context_name]
        return readout(session)

    # The page sits inside the project website, which carries the title. It is a grid of three
    # 288 pixel columns in two rows: the map, the agent, and the ground truth above three control
    # sections that read in the order a player uses them. State shows what the agent just did,
    # Move holds the loops that test memory ahead of the walks, and Settings holds what is
    # chosen once.
    with gr.Blocks(title="Context Compaction for World Models") as page:
        session_state = gr.State()
        with gr.Row(), gr.Column(scale=0, min_width=PAGE):
            with gr.Row():
                with gr.Column(scale=0, min_width=PANEL):
                    gr.Markdown("### Map")
                    map_image = room_image()
                with gr.Column(scale=0, min_width=PANEL):
                    agent_heading = gr.Markdown()
                    agent_image = room_image()
                with gr.Column(scale=0, min_width=PANEL):
                    gr.Markdown("### Ground Truth")
                    true_image = room_image()
            # Each control section sits in the theme's panel, its framed Windows 95 group box.
            with gr.Row(equal_height=True):
                with gr.Column(scale=0, min_width=PANEL, variant="panel"):
                    gr.Markdown("### State")
                    readout_text = gr.HTML()
                with gr.Column(scale=0, min_width=PANEL, variant="panel"):
                    gr.Markdown("### Move")
                    # Three loops to a row keep "Loop 128" on one line in a 288 pixel column.
                    with gr.Row():
                        loop_buttons = [
                            gr.Button(f"Loop {length}", min_width=0) for length in LOOP_LENGTHS[:3]
                        ]
                    with gr.Row():
                        loop_buttons += [
                            gr.Button(f"Loop {length}", min_width=0) for length in LOOP_LENGTHS[3:]
                        ]
                    with gr.Row():
                        walk_buttons = [
                            gr.Button(f"Random Walk {length}", min_width=0)
                            for length in WALK_LENGTHS
                        ]
                with gr.Column(scale=0, min_width=PANEL, variant="panel"):
                    gr.Markdown("### Settings")
                    # Each label sits beside its picker. Gradio merges neighbouring form fields
                    # into one block, so the checkbox sits in its own column as its own block.
                    with gr.Row(elem_classes="settings"):
                        # The model-label rule keeps the label to its text, so the picker takes
                        # the rest of the row.
                        gr.Markdown("Model", min_width=0, elem_classes="model-label")
                        model_name = gr.Dropdown(
                            list(MODELS),
                            value="CED",
                            show_label=False,
                            container=False,
                            scale=3,
                            min_width=0,
                            elem_classes="model-picker",
                        )
                    with gr.Row(elem_classes="settings"):
                        gr.Markdown("Context", min_width=0, elem_classes="model-label")
                        context_name = gr.Dropdown(
                            list(CONTEXTS),
                            value="Context Compaction",
                            show_label=False,
                            container=False,
                            scale=3,
                            min_width=0,
                            elem_classes="model-picker",
                        )
                    with gr.Column():
                        dreaming = gr.Checkbox(value=True, label="Dreaming", container=False)
                    new_world_button = gr.Button("New world")
        outputs = [
            session_state,
            map_image,
            agent_heading,
            agent_image,
            true_image,
            readout_text,
        ]
        # show_progress="hidden" keeps Gradio from outlining every output on each streamed step.
        page.load(
            lambda model, context: start(model, context, settings.seed),
            [model_name, context_name],
            outputs,
            show_progress="hidden",
        )
        for length, button in zip(LOOP_LENGTHS, loop_buttons, strict=True):
            button.click(
                move_handler(loop, length), [session_state], outputs, show_progress="hidden"
            )
        for length, button in zip(WALK_LENGTHS, walk_buttons, strict=True):
            button.click(
                move_handler(walk, length), [session_state], outputs, show_progress="hidden"
            )
        new_world_button.click(
            new_world, [session_state, model_name, context_name], outputs, show_progress="hidden"
        )
        model_name.change(
            select_model, [session_state, model_name], outputs, show_progress="hidden"
        )
        dreaming.change(set_dreaming, [session_state, dreaming], None)
        context_name.change(
            set_context, [session_state, context_name], [readout_text], show_progress="hidden"
        )
    return page


def theme() -> gr.themes.Base:
    """The Windows 95 theme with a navy fill behind a checked box's white tick."""
    windows = gr.Theme.from_hub(THEME)
    # The published theme draws the white tick on a white box, so a checked box looks empty.
    return windows.set(checkbox_background_color_selected="#000080")


def main(settings: PlaySettings) -> None:
    page = build(settings)
    page.launch(theme=theme(), css=CSS, inbrowser=settings.open_browser, footer_links=[])


if __name__ == "__main__":
    main(tyro.cli(PlaySettings))
