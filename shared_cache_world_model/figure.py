"""Draw revisit accuracy against gap length for every run under results/."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import tyro
from pydantic import BaseModel, DirectoryPath, Field

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FigureSettings(BaseModel):
    results: DirectoryPath = Field(default=Path("results"))
    experiment: str = Field(default="roomnav_sharedkv_v1", min_length=1)
    steps: int | None = Field(default=None, gt=0)
    run_names: tuple[str, ...] = Field(
        default=(), description="selected result directories, across any experiment"
    )
    metric: Literal["mean", "gain_mean"] = Field(default="mean")
    cache_boundary: int | None = Field(default=None, gt=0)
    title: str = Field(default="unseen generated rooms", min_length=1)
    out: Path = Field(default=Path("results/figure.png"))


def comparison_signature(summary: dict[str, object]) -> str:
    """The settings that must agree before two runs can share one figure."""
    config = summary["config"]
    if not isinstance(config, dict):
        raise TypeError("summary config must be a dictionary")
    model = config["model"]
    if not isinstance(model, dict):
        raise TypeError("summary model config must be a dictionary")
    treatment_fields = {"kind", "per_layer_window", "global_window"}
    model_without_treatment = {
        key: value for key, value in model.items() if key not in treatment_fields
    }
    recording = summary["recording"]
    probe_recording = summary["probe_recording"]
    if not isinstance(recording, dict) or not isinstance(probe_recording, dict):
        raise TypeError("summary recordings must be dictionaries")
    comparable = {
        "history": config["history"],
        "batch_size": config["batch_size"],
        "steps": config["steps"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "probe_bins": config["probe_bins"],
        "model": model_without_treatment,
        "recording": recording["sha256"],
        "probe_recording": probe_recording["sha256"],
    }
    return json.dumps(comparable, sort_keys=True)


def load_runs(
    results: Path,
    experiment: str = "roomnav_sharedkv_v1",
    steps: int | None = None,
    run_names: tuple[str, ...] = (),
) -> dict[str, list[dict[str, object]]]:
    """Probe reports grouped by model kind, one entry per run that has a probe."""
    runs: dict[str, list[dict[str, object]]] = defaultdict(list)
    seeds: dict[str, set[int]] = defaultdict(set)
    arm_models: dict[str, str] = {}
    expected_signature: str | None = None
    if run_names:
        summary_paths = [results / run_name / "summary.json" for run_name in run_names]
        missing = [path for path in summary_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"result has no summary: {missing[0]}")
    else:
        summary_paths = sorted(results.glob("*/summary.json"))
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text())
        if summary.get("probe") is None:
            continue
        config = summary["config"]
        if not isinstance(config, dict):
            continue
        if not run_names and config.get("experiment") != experiment:
            continue
        if steps is not None and config.get("steps") != steps:
            continue
        signature = comparison_signature(summary)
        if expected_signature is None:
            expected_signature = signature
        if signature != expected_signature:
            raise ValueError(f"run {summary_path.parent.name} does not match the experiment")
        model = config["model"]
        if not isinstance(model, dict):
            raise TypeError("summary model config must be a dictionary")
        kind = model["kind"]
        if not isinstance(kind, str):
            raise TypeError("summary model kind must be a string")
        if kind not in {"per_layer", "shared_cache", "ced", "ced_full"}:
            raise ValueError(f"run {summary_path.parent.name} has unknown model kind {kind!r}")
        arm_model = json.dumps(model, sort_keys=True)
        if kind in arm_models and arm_models[kind] != arm_model:
            raise ValueError(f"model {kind} runs do not use the same settings")
        arm_models[kind] = arm_model
        seed = config.get("seed")
        if not isinstance(seed, int):
            raise TypeError("summary seed must be an integer")
        if seed in seeds[kind]:
            raise ValueError(f"model {kind} has more than one result for seed {seed}")
        seeds[kind].add(seed)
        runs[kind].append(summary["probe"])
    if not runs:
        raise FileNotFoundError(f"no probe for experiment {experiment!r} under {results}")
    # Every arm in one figure must come from the same seeds, or a seed difference would read as
    # an architecture difference.
    seed_sets = [seeds[kind] for kind in sorted(runs)]
    if any(seed_set != seed_sets[0] for seed_set in seed_sets):
        raise ValueError("the model kinds do not contain the same seeds")
    return runs


def bin_start(label: str) -> int:
    return int(label.split("-")[0].rstrip("+"))


def curve(
    reports: list[dict[str, object]], metric: str = "mean"
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Mean and standard deviation over runs of the per-bin mean accuracy, ordered by gap."""
    per_bin: dict[str, list[float]] = defaultdict(list)
    expected_labels: set[str] | None = None
    for report in reports:
        labels = set(report)
        if expected_labels is None:
            expected_labels = labels
        if labels != expected_labels:
            raise ValueError("runs do not contain the same gap bins")
        for label, summary in report.items():
            if not isinstance(summary, dict):
                raise TypeError(f"gap bin {label!r} must be a dictionary")
            per_bin[label].append(float(summary[metric]))
    labels = sorted(per_bin, key=bin_start)
    means = np.array([np.mean(per_bin[label]) for label in labels])
    spreads = np.array([np.std(per_bin[label]) for label in labels])
    return [bin_start(label) for label in labels], means, spreads


def main(settings: FigureSettings) -> None:
    runs = load_runs(
        Path(settings.results), settings.experiment, settings.steps, settings.run_names
    )
    plot = plt.subplots(figsize=(6, 4))
    figure = plot[0]
    axis = plot[1]
    # The paper calls the windowed variant CED-SWA and the faithful variant CED.
    display_names = {
        "per_layer": "per-layer",
        "shared_cache": "shared cache",
        "ced": "CED-SWA",
        "ced_full": "CED",
    }
    for kind, reports in sorted(runs.items()):
        metric_curve = curve(reports, metric=settings.metric)
        starts = metric_curve[0]
        means = metric_curve[1]
        spreads = metric_curve[2]
        axis.errorbar(
            starts,
            means,
            yerr=spreads,
            marker="o",
            capsize=3,
            label=f"{display_names[kind]} ({len(reports)} runs)",
        )
        if settings.metric == "mean":
            shuffled_curve = curve(reports, metric="shuffled_mean")
            shuffled_starts = shuffled_curve[0]
            shuffled_means = shuffled_curve[1]
            axis.plot(
                shuffled_starts,
                shuffled_means,
                marker="o",
                linestyle=":",
                label=f"{display_names[kind]} shuffled history",
            )
    if settings.cache_boundary is not None:
        axis.axvline(settings.cache_boundary, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("steps since the room was last visited")
    if settings.metric == "gain_mean":
        axis.set_ylabel("causal retrieval gain")
        axis.set_ylim(-0.05, 0.7)
    else:
        axis.set_ylabel("background tiles correct on re-entry")
        axis.set_ylim(0, 1)
    axis.set_title(settings.title)
    axis.legend()
    figure.tight_layout()
    settings.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(settings.out, dpi=150)
    logger.info("wrote %s", settings.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(tyro.cli(FigureSettings))
