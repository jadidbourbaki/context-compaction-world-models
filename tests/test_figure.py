from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared_cache_world_model.figure import FigureSettings, curve, load_runs, main


def write_summary(
    results: Path,
    run: str,
    kind: str,
    accuracies: dict[str, float],
    seed: int = 42,
    experiment: str = "roomnav_sharedkv_v1",
    per_layer_window: int | None = None,
    global_window: int | None = None,
) -> None:
    run_dir = results / run
    run_dir.mkdir(parents=True)
    probe = {
        label: {
            "mean": value,
            "shuffled_mean": value / 2,
            "gain_mean": value / 2,
            "n": 3,
        }
        for label, value in accuracies.items()
    }
    model = {"kind": kind, "d_model": 32, "n_heads": 4, "n_layers": 4}
    if per_layer_window is not None:
        model["per_layer_window"] = per_layer_window
    if global_window is not None:
        model["global_window"] = global_window
    config = {
        "experiment": experiment,
        "seed": seed,
        "model": model,
        "history": 16,
        "batch_size": 2,
        "steps": 10,
        "lr": 0.001,
        "weight_decay": 0.1,
        "probe_bins": [1, 2, 4, 8, 16],
    }
    summary = {
        "config": config,
        "probe": probe,
        "recording": {"sha256": "train"},
        "probe_recording": {"sha256": "eval"},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_curve_averages_runs_and_orders_bins_by_gap(tmp_path: Path) -> None:
    write_summary(tmp_path, "a", "per_layer", {"16-31": 0.8, "2-3": 1.0, "128-255": 0.4})
    write_summary(
        tmp_path,
        "b",
        "per_layer",
        {"16-31": 0.6, "2-3": 1.0, "128-255": 0.2},
        seed=43,
    )
    write_summary(tmp_path, "c", "shared_cache", {"2-3": 0.9})
    write_summary(tmp_path, "d", "shared_cache", {"2-3": 0.8}, seed=43)
    runs = load_runs(tmp_path)
    assert sorted(runs) == ["per_layer", "shared_cache"]
    per_layer_curve = curve(runs["per_layer"])
    starts = per_layer_curve[0]
    means = per_layer_curve[1]
    spreads = per_layer_curve[2]
    assert starts == [2, 16, 128]
    assert np.allclose(means, [1.0, 0.7, 0.3])
    assert np.allclose(spreads, [0.0, 0.1, 0.1])


def test_main_writes_a_png(tmp_path: Path) -> None:
    write_summary(tmp_path, "a", "per_layer", {"2-3": 1.0, "16-31": 0.5})
    out = tmp_path / "figure.png"
    main(FigureSettings(results=tmp_path, out=out))
    assert out.exists()


def test_main_selects_runs_across_experiments_and_plots_gain(tmp_path: Path) -> None:
    write_summary(
        tmp_path,
        "per",
        "per_layer",
        {"2-3": 0.8, "16-31": 0.4},
        experiment="bounded",
        per_layer_window=2,
    )
    write_summary(
        tmp_path,
        "shared",
        "shared_cache",
        {"2-3": 0.9, "16-31": 0.8},
        experiment="full",
    )
    out = tmp_path / "gain.png"
    main(
        FigureSettings(
            results=tmp_path,
            run_names=("per", "shared"),
            metric="gain_mean",
            cache_boundary=2,
            out=out,
        )
    )
    assert out.exists()


def test_load_runs_rejects_unpaired_seeds(tmp_path: Path) -> None:
    write_summary(tmp_path, "a", "per_layer", {"16-31": 0.8})
    write_summary(tmp_path, "b", "shared_cache", {"16-31": 0.8}, seed=43)
    with pytest.raises(ValueError, match="same seeds"):
        load_runs(tmp_path)


def test_load_runs_groups_three_arms_and_ignores_their_windows(tmp_path: Path) -> None:
    # The windows are treatments, so arms with different windows still belong to one figure.
    write_summary(tmp_path, "a", "per_layer", {"16-31": 0.8}, per_layer_window=2)
    write_summary(tmp_path, "b", "shared_cache", {"16-31": 0.6})
    write_summary(tmp_path, "c", "ced", {"16-31": 0.7}, global_window=4)
    runs = load_runs(tmp_path)
    assert sorted(runs) == ["ced", "per_layer", "shared_cache"]
    write_summary(tmp_path, "d", "ced", {"16-31": 0.7}, seed=43, global_window=4)
    with pytest.raises(ValueError, match="same seeds"):
        load_runs(tmp_path)


def test_load_runs_selects_training_step_count(tmp_path: Path) -> None:
    write_summary(tmp_path, "pilot", "per_layer", {"16-31": 0.5})
    write_summary(tmp_path, "batch", "shared_cache", {"16-31": 0.8})
    batch_path = tmp_path / "batch" / "summary.json"
    batch = json.loads(batch_path.read_text())
    batch["config"]["steps"] = 20
    batch_path.write_text(json.dumps(batch))
    runs = load_runs(tmp_path, steps=20)
    assert sorted(runs) == ["shared_cache"]


def test_curve_rejects_runs_with_different_gap_bins(tmp_path: Path) -> None:
    write_summary(tmp_path, "a", "per_layer", {"2-3": 1.0, "4-7": 0.5})
    write_summary(tmp_path, "b", "per_layer", {"2-3": 1.0}, seed=43)
    runs = load_runs(tmp_path)
    with pytest.raises(ValueError, match="same gap bins"):
        curve(runs["per_layer"])
