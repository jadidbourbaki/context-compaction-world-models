"""Render docs/demo.gif, a CED world agent closing two loops with compaction and dreaming."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from shared_cache_world_model.agent import plan_loop
from shared_cache_world_model.play import (
    CONTEXTS,
    MODELS,
    PANEL,
    STEP_SECONDS,
    Session,
    load_run,
    render_map,
)
from shared_cache_world_model.world import add_character, render_room

OUT = Path(__file__).parent / "demo.gif"
SEED = 43
LOOPS = (16, 64)
GAP = 12
HEADER = 32
HOLD_SECONDS = 2.0
BACKGROUND = (255, 255, 255)
INK = (0, 0, 0)


def frame(session: Session, font: ImageFont.FreeTypeFont) -> Image.Image:
    """The map, the agent's room, and the true room side by side under their titles."""
    remembered = session.record.remembered if session.record is not None else None
    panels = [
        ("Map", render_map(session)),
        ("Agent", render_room(add_character(session.agent.view.numpy()))),
        (
            "Ground Truth",
            render_room(add_character(session.agent.true_room(session.agent.room).numpy())),
        ),
    ]
    width = 3 * PANEL + 4 * GAP
    height = HEADER + PANEL + GAP
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index, (title, pixels) in enumerate(panels):
        left = GAP + index * (PANEL + GAP)
        draw.text((left, HEADER // 2), title, fill=INK, font=font, anchor="lm")
        image.paste(Image.fromarray(pixels), (left, HEADER))
    # A predicted revisit shows its accuracy at the right of the agent's title.
    if remembered is not None:
        right = 2 * (PANEL + GAP)
        accuracy = f"{100 * remembered.accuracy:.1f}%"
        draw.text((right, HEADER // 2), accuracy, fill=INK, font=font, anchor="rm")
    return image


def main() -> None:
    device = torch.accelerator.current_accelerator()
    if device is None:
        raise RuntimeError("no accelerator found")
    model, state = load_run(MODELS["CED"], str(device))
    session = Session(model, state, SEED, CONTEXTS["Context Compaction"])
    font = ImageFont.load_default(size=18)
    frames = [frame(session, font)]
    durations = [HOLD_SECONDS]
    for loop_steps in LOOPS:
        previous_action = session.agent.actions[-1] if session.agent.actions else None
        session.queued = plan_loop(
            session.world_settings, session.agent.room, loop_steps, previous_action, session.rng
        )
        while session.queued:
            session.record = session.agent.step(session.queued.pop(0))
            frames.append(frame(session, font))
            durations.append(STEP_SECONDS)
        durations[-1] = HOLD_SECONDS
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
