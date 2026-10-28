"""Run the closed-loop agent on every same-context checkpoint for Experiment 4."""

# Every run plays the same worlds, drawn from the evaluation seed, so the conditions compare
# paired revisits. A condition is one checkpoint, one context, dreaming on or off, and one loop
# length. Each run writes only its summaries, because every step of a batch would fill 170 MB.

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import tyro
from pydantic import BaseModel, DirectoryPath, Field

from shared_cache_world_model.agent import AgentSettings, ContextMode
from shared_cache_world_model.agent import main as run_agent

logger = logging.getLogger(__name__)

KINDS = ("per_layer", "ced_full", "ced", "shared_cache")
SEEDS = (42, 43, 44)
CONTEXTS: tuple[ContextMode, ...] = ("raw", "compacted")
LOOP_LENGTHS = (8, 32, 128)


class BatchSettings(BaseModel):
    weights: DirectoryPath = Field(default=Path("weights"))
    results: DirectoryPath = Field(default=Path("results"))
    steps: int = Field(default=512, ge=1, description="room transitions per episode")
    # A 512-step episode holds 32 loops of 8 steps but only 2 of 128, so the longest loop runs
    # twice the episodes to close 40 loops per run.
    episodes: int = Field(default=10, ge=1)
    long_loop_episodes: int = Field(default=20, ge=1)
    # Four processes share the laptop GPU at about 2.8 times the throughput of one.
    workers: int = Field(default=4, ge=1)


def batch_runs(settings: BatchSettings) -> list[AgentSettings]:
    """The settings of every run in the batch."""
    runs = []
    conditions = itertools.product(KINDS, SEEDS, CONTEXTS, (True, False), LOOP_LENGTHS)
    for condition in conditions:
        kind = condition[0]
        seed = condition[1]
        context = condition[2]
        dreaming = condition[3]
        loop_steps = condition[4]
        run = f"roomnav_transition_v1_{kind}_h128_s{seed}"
        dreaming_name = "dreaming" if dreaming else "no-dreaming"
        out = settings.results / run / "agent" / f"loop{loop_steps}_{context}_{dreaming_name}.json"
        episodes = settings.long_loop_episodes if loop_steps == 128 else settings.episodes
        runs.append(
            AgentSettings(
                checkpoint=settings.weights / run / "weights.pt",
                context=context,
                dreaming=dreaming,
                episodes=episodes,
                steps=settings.steps,
                policy="loop",
                loop_steps=loop_steps,
                out=out,
                save_records=False,
            )
        )
    return runs


def run_logged(run: AgentSettings) -> None:
    """Run one agent in a worker process with its own log format."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_agent(run)


def main(settings: BatchSettings) -> None:
    runs = batch_runs(settings)
    logger.info("%d runs on %d workers", len(runs), settings.workers)
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        # list() waits for every run and raises the first failure.
        list(pool.map(run_logged, runs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(BatchSettings))
