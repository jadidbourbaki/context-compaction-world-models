"""Render docs/demo.gif, two CED world agents walking a loop longer than the raw window."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from shared_cache_world_model.agent import StepRecord, plan_loop
from shared_cache_world_model.play import (
    CONTEXTS,
    MODELS,
    PANEL,
    Session,
    load_run,
    render_map,
)
from shared_cache_world_model.world import add_character, render_room

OUT = Path(__file__).parent / "demo.gif"
SEED = 43
# The loop returns to its start room 192 steps later, past the 128-step raw window.
LOOP_STEPS = 192
GAP = 12
HEADER = 32
STEP_SECONDS = 0.06
START_SECONDS = 1.0
END_SECONDS = 4.0
BACKGROUND = (255, 255, 255)
INK = (0, 0, 0)


def room_pixels(session: Session) -> Image.Image:
    """The room the agent holds for its current room, with the character."""
    return Image.fromarray(render_room(add_character(session.agent.view.numpy())))


def frame(
    raw: Session,
    compacted: Session,
    records: tuple[StepRecord | None, StepRecord | None],
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """The map, the raw window agent, the compaction agent, and the true room under titles."""
    truth = render_room(add_character(raw.agent.true_room(raw.agent.room).numpy()))
    panels = [
        ("Map", Image.fromarray(render_map(raw)), None),
        ("Raw Window", room_pixels(raw), records[0]),
        ("Context Compaction", room_pixels(compacted), records[1]),
        ("Ground Truth", Image.fromarray(truth), None),
    ]
    width = len(panels) * PANEL + (len(panels) + 1) * GAP
    height = HEADER + PANEL + GAP
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index, (title, pixels, record) in enumerate(panels):
        left = GAP + index * (PANEL + GAP)
        draw.text((left, HEADER // 2), title, fill=INK, font=font, anchor="lm")
        image.paste(pixels, (left, HEADER))
        # A predicted revisit shows its accuracy at the right of the agent's title.
        if record is not None and record.remembered is not None:
            accuracy = f"{100 * record.remembered.accuracy:.0f}%"
            draw.text((left + PANEL, HEADER // 2), accuracy, fill=INK, font=font, anchor="rm")
    return image


def main() -> None:
    device = torch.accelerator.current_accelerator()
    if device is None:
        raise RuntimeError("no accelerator found")
    model, state = load_run(MODELS["CED"], str(device))
    # Both agents see the same world from the same start room and take the same actions.
    raw = Session(model, state, SEED, CONTEXTS["Raw Window"])
    compacted = Session(model, state, SEED, CONTEXTS["Context Compaction"])
    font = ImageFont.load_default(size=18)
    actions = plan_loop(raw.world_settings, raw.agent.room, LOOP_STEPS, None, raw.rng)
    raw.queued = list(actions)
    records: tuple[StepRecord | None, StepRecord | None] = (None, None)
    frames = [frame(raw, compacted, records, font)]
    durations = [START_SECONDS]
    for action in actions:
        raw.queued.pop(0)
        records = (raw.agent.step(action), compacted.agent.step(action))
        frames.append(frame(raw, compacted, records, font))
        durations.append(STEP_SECONDS)
    durations[-1] = END_SECONDS
    for name, record in zip(("raw window", "context compaction"), records, strict=True):
        if record is None or record.remembered is None:
            raise RuntimeError("the loop must end on a revisit")
        print(f"{name}: gap {record.gap}, accuracy {record.remembered.accuracy:.3f}")
    # The sprites and the route use a handful of colours, so an adaptive palette keeps them exact.
    palette_frames = [image.quantize(colors=16, dither=Image.Dither.NONE) for image in frames]
    palette_frames[0].save(
        OUT,
        save_all=True,
        append_images=palette_frames[1:],
        duration=[int(1000 * seconds) for seconds in durations],
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUT} with {len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
