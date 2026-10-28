# Reproducing the paper

Run every command from the repository root. `results/` already holds every result, so the
figure commands work right after setup.

## Data

Generate the training recording.

```sh
just world --out data/roomnav_train --episodes 5000 --seed 42
```

Generate the evaluation recording.

```sh
just world --out data/roomnav_eval --episodes 400 --seed 43
```

## Weights

Download the 21 trained world models.

```sh
just fetch-weights
```

## Training

Train one same-context world model. Run it for each `KIND` in `per_layer`, `ced_full`, `ced`,
and `shared_cache`, and each `SEED` in 42, 43, and 44.

```sh
just train --recording data/roomnav_train --probe-recording data/roomnav_eval --experiment roomnav_transition_v1 --model.kind KIND --steps 20000 --probe-every 5000 --seed SEED
```

Train the bounded Decoder-only world model for each seed.

```sh
just train --recording data/roomnav_train --probe-recording data/roomnav_eval --experiment roomnav_equal_memory_v1 --model.kind per_layer --model.per-layer-window 16 --steps 20000 --probe-every 5000 --seed SEED --run-name roomnav_equal_memory_v1_per_layer_w16_h128_sSEED
```

Train the bounded CED world model for each seed.

```sh
just train --recording data/roomnav_train --probe-recording data/roomnav_eval --experiment roomnav_equal_memory_v1 --model.kind ced_full --model.per-layer-window 16 --model.global-window 16 --steps 20000 --probe-every 5000 --seed SEED --run-name roomnav_equal_memory_v1_ced_full_w16_g16_h128_sSEED
```

Train the bounded CED-SWA world model for each seed.

```sh
just train --recording data/roomnav_train --probe-recording data/roomnav_eval --experiment roomnav_equal_memory_v1 --model.kind ced --model.global-window 32 --steps 20000 --probe-every 5000 --seed SEED --run-name roomnav_equal_memory_v1_ced_g32_h128_sSEED
```

Export the trained checkpoints to `weights/`.

```sh
just export-weights --out weights
```

## Evaluation

Score one world model through one context. Run it for each `CONTEXT` in `compacted`,
`compacted_when_full`, and `truncated`.

```sh
just probe --checkpoint weights/roomnav_transition_v1_KIND_h128_sSEED/weights.pt --model.kind KIND --recording data/roomnav_eval --training-recording data/roomnav_train --context-mode CONTEXT --out results/roomnav_transition_v1_KIND_h128_sSEED/CONTEXT
```

Run every world agent with and without dreaming.

```sh
just agent-batch
```

Time decode and prefill.

```sh
just systems
```

Time every step of one session.

```sh
uv run python -m shared_cache_world_model.session
```

## Figures

Draw Figure 1.

```sh
uv run python paper/make_teaser_figure.py
```

Draw Figure 2.

```sh
uv run python paper/make_benchmark_figure.py
```

Draw Figure 5.

```sh
uv run python paper/make_interaction_figure.py
```

Draw Figures 6 and 7.

```sh
uv run python paper/make_result_figure.py
```

Draw Figure 8.

```sh
uv run python paper/make_revisit_figure.py
```

Draw Figure 9.

```sh
uv run python paper/make_latency_figure.py
```

Draw Figure 10.

```sh
uv run python paper/make_compaction_figure.py
```

Draw Figure 11.

```sh
uv run python paper/make_contexts_figure.py
```

Draw Figure 12.

```sh
uv run python paper/make_agent_figure.py
```

Table 1 comes from `results/systems/summary.json`. Figures 3 and 4 are drawn in the paper
repository.
