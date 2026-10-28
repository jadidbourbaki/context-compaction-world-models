from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from torch.nn.attention.flex_attention import flex_attention

from shared_cache_world_model.data import TILES_PER_FRAME
from shared_cache_world_model.model import (
    TOKENS_PER_STEP,
    ModelSettings,
    ProjectedMemory,
    WorldModel,
    attend,
    block_mask,
    dense_mask,
    masked_mean,
    next_frame_losses,
    tile_target_mask,
)
from shared_cache_world_model.world import ACTIONS, ROOM_COLS, ROOM_ROWS, TILE_VOCAB

KINDS = ["per_layer", "shared_cache", "ced", "ced_full"]
# Every arm whose upper blocks read a projected global cache through cross-attention.
CROSS_KINDS = ["shared_cache", "ced", "ced_full"]
# The cross-attention arms whose lower stack is bounded to two steps.
WINDOWED_LOWER_KINDS = ["shared_cache", "ced"]
# The arms that give every upper block its own projected pair.
CED_KINDS = ["ced", "ced_full"]


def tiny(kind: str, seed: int = 42, global_window: int | None = None) -> WorldModel:
    torch.manual_seed(seed)
    settings = ModelSettings(
        kind=kind,  # type: ignore[arg-type]
        d_model=32,
        n_heads=4,
        n_layers=4,
        max_history=6,
        global_window=global_window,
    )
    model = WorldModel(settings)
    model.eval()
    return model


def test_model_settings_reject_odd_rotary_head_width() -> None:
    with pytest.raises(ValueError, match="must be even"):
        ModelSettings(kind="per_layer", d_model=12, n_heads=4, n_layers=2)


@pytest.mark.parametrize("kind", WINDOWED_LOWER_KINDS)
def test_model_settings_reject_per_layer_window_on_windowed_lower_arms(kind: str) -> None:
    with pytest.raises(ValueError, match="applies only to the per_layer and ced_full"):
        ModelSettings(kind=kind, per_layer_window=16)  # type: ignore[arg-type]


def test_model_settings_reject_global_window_on_per_layer() -> None:
    with pytest.raises(ValueError, match="applies only to the shared_cache and ced"):
        ModelSettings(kind="per_layer", global_window=16)


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("per_layer", "per_layer_window"),
        ("shared_cache", "global_window"),
        ("ced", "global_window"),
    ],
)
def test_model_settings_reject_window_beyond_maximum_history(kind: str, field: str) -> None:
    with pytest.raises(ValueError, match="exceeds max_history"):
        ModelSettings(kind=kind, max_history=8, **{field: 16})  # type: ignore[arg-type]


def sample(batch: int = 2, steps: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, TILE_VOCAB, (batch, steps, ROOM_ROWS, ROOM_COLS), generator=generator)
    actions = torch.randint(0, len(ACTIONS), (batch, steps), generator=generator)
    return frames, actions


def test_input_tokens_hold_eight_tile_masks_and_one_action() -> None:
    model = tiny("per_layer")
    inputs = sample(batch=1, steps=2)
    frames = inputs[0]
    actions = inputs[1]
    tokens = model.input_tokens(frames, actions)
    assert tokens.shape == (1, 2, TOKENS_PER_STEP, model.settings.d_model)
    expected_action = model.token_embedding(actions + TILE_VOCAB)
    torch.testing.assert_close(tokens[:, :, -1], expected_action)


@given(
    window=st.one_of(st.none(), st.integers(min_value=1, max_value=4)),
    steps=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@hypothesis_settings(max_examples=8, deadline=None)
def test_block_mask_matches_the_dense_mask(window: int | None, steps: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    length = steps * TOKENS_PER_STEP
    q = torch.randn(1, 2, length, 8, generator=generator)
    k = torch.randn(1, 2, length, 8, generator=generator)
    v = torch.randn(1, 2, length, 8, generator=generator)
    device = torch.device("cpu")
    with torch.no_grad():
        # The GPU path's block mask and the CPU path's dense mask come from the same rule. Eager
        # FlexAttention runs forward on the CPU, so the two paths are compared here.
        flex = flex_attention(q, k, v, block_mask=block_mask(length, window, device))
        dense = attend(q, k, v, window)
    torch.testing.assert_close(flex, dense, atol=1e-4, rtol=1e-4)
    visible = dense_mask(length, window, device)
    positions = torch.arange(length)
    query_steps = positions[:, None] // TOKENS_PER_STEP
    key_steps = positions[None, :] // TOKENS_PER_STEP
    expected = key_steps <= query_steps
    if window is not None:
        expected = expected & (query_steps - key_steps < window)
    assert torch.equal(visible, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_forward_shapes(kind: str) -> None:
    model = tiny(kind)
    inputs = sample()
    frames = inputs[0]
    actions = inputs[1]
    with torch.no_grad():
        logits = model(frames, actions)
    assert logits.shape == (2, 5, TILE_VOCAB, TILES_PER_FRAME)


@given(
    kind=st.sampled_from(KINDS),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps=st.integers(min_value=2, max_value=5),
)
@hypothesis_settings(max_examples=12, deadline=None)
def test_no_step_sees_a_later_step(kind: str, seed: int, steps: int) -> None:
    model = tiny(kind, seed=seed)
    inputs = sample(batch=1, steps=steps, seed=seed)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    changed[0, steps - 1] = (changed[0, steps - 1] + 1) % TILE_VOCAB
    with torch.no_grad():
        before = model(frames, actions)
        after = model(changed, actions)
    torch.testing.assert_close(before[:, : steps - 1], after[:, : steps - 1])
    assert not torch.allclose(before[:, steps - 1 :], after[:, steps - 1 :])


@given(
    kind=st.sampled_from(KINDS),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    row=st.integers(min_value=0, max_value=ROOM_ROWS - 1),
    col=st.integers(min_value=0, max_value=ROOM_COLS - 1),
)
@hypothesis_settings(max_examples=12, deadline=None)
def test_every_token_of_a_step_sees_the_whole_step(
    kind: str, seed: int, row: int, col: int
) -> None:
    model = tiny(kind, seed=seed)
    inputs = sample(batch=1, steps=3, seed=seed)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    original_tile = int(changed[0, 1, row, col])
    changed[0, 1, row, col] = (changed[0, 1, row, col] + 1) % TILE_VOCAB
    unaffected_tile = (original_tile + 2) % TILE_VOCAB
    with torch.no_grad():
        before = model(frames, actions)
        after = model(changed, actions)
    # The unchanged tile token can react only by attending to the changed tokens in its step.
    assert not torch.allclose(before[:, 1, unaffected_tile], after[:, 1, unaffected_tile])


@pytest.mark.parametrize("kind", WINDOWED_LOWER_KINDS)
def test_cross_attention_lower_stack_sees_two_steps_only(kind: str) -> None:
    model = tiny(kind)
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    changed[0, 0] = (changed[0, 0] + 1) % TILE_VOCAB
    with torch.no_grad():
        x = model.embed(frames, actions)
        x_changed = model.embed(changed, actions)
        for block in model.lower:
            x = block(x, window=2)
            x_changed = block(x_changed, window=2)
    # Each of the two lower blocks reaches one step back, so step 0 influences steps 0 to 2 and
    # step 3 never sees it. The receptive field grows by one step per local block.
    boundary = 3 * TOKENS_PER_STEP
    torch.testing.assert_close(x[:, boundary:], x_changed[:, boundary:])
    assert not torch.allclose(x[:, :boundary], x_changed[:, :boundary])


def test_ced_full_lower_stack_reads_the_whole_history() -> None:
    model = tiny("ced_full")
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    changed[0, 0] = (changed[0, 0] + 1) % TILE_VOCAB
    with torch.no_grad():
        x = model.embed(frames, actions)
        x_changed = model.embed(changed, actions)
        for block in model.lower:
            x = block(x, window=model.lower_window())
            x_changed = block(x_changed, window=model.lower_window())
    # Ordinary lower blocks attend over every earlier step, so step 0 reaches the last step.
    boundary = 3 * TOKENS_PER_STEP
    assert not torch.allclose(x[:, boundary:], x_changed[:, boundary:])


def test_ced_full_matches_per_layer_parameters_up_to_the_shared_norm() -> None:
    # Four ordinary lower blocks plus four cross-attention blocks with their own key and value
    # projections carry the same weights as eight self-attention blocks. The one extra piece is
    # the normalization applied to the lower output before it is projected.
    per_layer_model = tiny("per_layer")
    ced_full_model = tiny("ced_full")
    per_layer_count = sum(parameter.numel() for parameter in per_layer_model.parameters())
    ced_full_count = sum(parameter.numel() for parameter in ced_full_model.parameters())
    # A LayerNorm holds a scale and a shift per model dimension.
    assert ced_full_count - per_layer_count == 2 * ced_full_model.settings.d_model


@pytest.mark.parametrize("kind", CROSS_KINDS)
def test_cross_attention_upper_stack_reads_the_whole_history(kind: str) -> None:
    model = tiny(kind)
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    changed[0, 0] = (changed[0, 0] + 1) % TILE_VOCAB
    with torch.no_grad():
        before = model(frames, actions)
        after = model(changed, actions)
    # The final step's logits depend on step 0 through the global cache.
    assert not torch.allclose(before[:, 3:], after[:, 3:])


@pytest.mark.parametrize("kind", WINDOWED_LOWER_KINDS)
def test_windowed_cross_attention_cannot_cross_its_stacked_receptive_field(kind: str) -> None:
    # Two lower blocks with a two-step window reach two steps back, so the normalized source at
    # step t holds steps t - 2 through t. A two-step global window reads the source at steps
    # t - 1 and t, so an upper block at step t reaches step t - 3 and no further. Step 0 can
    # therefore influence steps 0 through 3 and never step 4.
    model = tiny(kind, global_window=2)
    inputs = sample(batch=1, steps=5)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    changed[0, 0] = (changed[0, 0] + 1) % TILE_VOCAB
    with torch.no_grad():
        before = model(frames, actions)
        after = model(changed, actions)
    torch.testing.assert_close(before[:, 4], after[:, 4])
    assert not torch.allclose(before[:, 3], after[:, 3])


def test_windowed_per_layer_model_cannot_cross_its_stacked_receptive_field() -> None:
    torch.manual_seed(42)
    model = WorldModel(
        ModelSettings(
            kind="per_layer",
            d_model=32,
            n_heads=4,
            n_layers=2,
            max_history=4,
            per_layer_window=2,
        )
    )
    model.eval()
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    changed = frames.clone()
    changed[0, 0] = (changed[0, 0] + 1) % TILE_VOCAB
    with torch.no_grad():
        before = model(frames, actions)
        after = model(changed, actions)
    torch.testing.assert_close(before[:, 3], after[:, 3])
    assert not torch.allclose(before[:, 2], after[:, 2])


def test_shared_cache_projects_one_global_kv_pair() -> None:
    model = tiny("shared_cache")
    if model.shared_kv is None:
        raise AssertionError("shared-cache model has no global projection")
    projection_calls = 0
    memories: list[object] = []

    def count_projection(
        _module: torch.nn.Module, _inputs: tuple[object, ...], _output: object
    ) -> None:
        nonlocal projection_calls
        projection_calls += 1

    def record_memory(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        memories.append(inputs[1])

    handles = [model.shared_kv.register_forward_hook(count_projection)]
    for block in model.upper:
        if block.cross_attention is None:
            raise AssertionError("shared-cache upper block has no cross-attention")
        handles.append(block.cross_attention.register_forward_pre_hook(record_memory))
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    with torch.no_grad():
        model(frames, actions)
    for handle in handles:
        handle.remove()
    assert projection_calls == 1
    assert len(memories) == len(model.upper)
    assert all(memory is memories[0] for memory in memories)


@pytest.mark.parametrize("kind", CED_KINDS)
def test_ced_projects_one_kv_pair_per_upper_block(kind: str) -> None:
    model = tiny(kind)
    projection_calls = 0
    memories: list[ProjectedMemory] = []

    def count_projection(
        _module: torch.nn.Module, _inputs: tuple[object, ...], _output: object
    ) -> None:
        nonlocal projection_calls
        projection_calls += 1

    def record_memory(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        memory = inputs[1]
        if not isinstance(memory, ProjectedMemory):
            raise AssertionError("cross-attention received something other than a memory")
        memories.append(memory)

    handles = [projection.register_forward_hook(count_projection) for projection in model.upper_kv]
    for block in model.upper:
        if block.cross_attention is None:
            raise AssertionError("ced upper block has no cross-attention")
        handles.append(block.cross_attention.register_forward_pre_hook(record_memory))
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    with torch.no_grad():
        model(frames, actions)
    for handle in handles:
        handle.remove()
    # Every upper block projects its own pair from the one shared source, so the pairs differ.
    assert projection_calls == len(model.upper)
    assert len(memories) == len(model.upper)
    assert len({id(memory) for memory in memories}) == len(memories)
    assert not torch.allclose(memories[0].key, memories[1].key)
    assert not torch.allclose(memories[0].value, memories[1].value)


@given(
    kind=st.sampled_from(KINDS),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps=st.integers(min_value=1, max_value=5),
)
@hypothesis_settings(max_examples=12, deadline=None)
def test_incremental_cache_matches_the_full_window(kind: str, seed: int, steps: int) -> None:
    model = tiny(kind, seed=seed)
    inputs = sample(batch=1, steps=steps, seed=seed)
    frames = inputs[0]
    actions = inputs[1]
    cache = None
    incremental_logits = []
    with torch.no_grad():
        full_logits = model(frames, actions)
        for step in range(actions.shape[1]):
            result = model.forward_step(frames[:, step], actions[:, step], cache)
            incremental_logits.append(result.logits)
            cache = result.cache
    joined_logits = torch.cat(incremental_logits, dim=1)
    torch.testing.assert_close(joined_logits, full_logits, atol=1e-5, rtol=1e-5)


@given(
    kind=st.sampled_from(KINDS),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps=st.integers(min_value=1, max_value=6),
    global_window=st.sampled_from([None, 2, 3]),
)
@hypothesis_settings(max_examples=16, deadline=None)
def test_last_step_prefill_matches_the_full_forward(
    kind: str, seed: int, steps: int, global_window: int | None
) -> None:
    model = tiny(kind, seed=seed, global_window=None if kind == "per_layer" else global_window)
    inputs = sample(batch=2, steps=steps, seed=seed)
    frames = inputs[0]
    actions = inputs[1]
    with torch.no_grad():
        full_logits = model(frames, actions)
        last_logits = model.forward_last(frames, actions)
    torch.testing.assert_close(last_logits, full_logits[:, -1], atol=1e-5, rtol=1e-5)


# ced_full bounded like the per-layer model windows its lower blocks and its upper reads alike.
@pytest.mark.parametrize("kind", ["per_layer", "ced_full"])
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps=st.integers(min_value=3, max_value=6),
)
@hypothesis_settings(max_examples=6, deadline=None)
def test_windowed_self_attention_cache_matches_the_full_window(
    kind: str, seed: int, steps: int
) -> None:
    torch.manual_seed(seed)
    model = WorldModel(
        ModelSettings(
            kind=kind,  # type: ignore[arg-type]
            d_model=32,
            n_heads=4,
            n_layers=4,
            max_history=6,
            per_layer_window=2,
            global_window=None if kind == "per_layer" else 2,
        )
    )
    model.eval()
    inputs = sample(batch=1, steps=steps, seed=seed)
    frames = inputs[0]
    actions = inputs[1]
    cache = None
    incremental_logits = []
    with torch.no_grad():
        full_logits = model(frames, actions)
        for step in range(steps):
            result = model.forward_step(frames[:, step], actions[:, step], cache)
            incremental_logits.append(result.logits)
            cache = result.cache
    joined_logits = torch.cat(incremental_logits, dim=1)
    torch.testing.assert_close(joined_logits, full_logits, atol=1e-5, rtol=1e-5)
    if cache is None:
        raise AssertionError("incremental model produced no cache")
    retained_lengths = [memory.key.shape[2] for memory in cache.layers if memory is not None]
    assert retained_lengths
    assert all(length == TOKENS_PER_STEP for length in retained_lengths)


@given(
    kind=st.sampled_from(CROSS_KINDS),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps=st.integers(min_value=3, max_value=6),
)
@hypothesis_settings(max_examples=8, deadline=None)
def test_windowed_cross_attention_cache_matches_the_full_window(
    kind: str, seed: int, steps: int
) -> None:
    model = tiny(kind, seed=seed, global_window=2)
    inputs = sample(batch=1, steps=steps, seed=seed)
    frames = inputs[0]
    actions = inputs[1]
    cache = None
    incremental_logits = []
    with torch.no_grad():
        full_logits = model(frames, actions)
        for step in range(steps):
            result = model.forward_step(frames[:, step], actions[:, step], cache)
            incremental_logits.append(result.logits)
            cache = result.cache
    joined_logits = torch.cat(incremental_logits, dim=1)
    torch.testing.assert_close(joined_logits, full_logits, atol=1e-5, rtol=1e-5)
    if cache is None:
        raise AssertionError("incremental model produced no cache")
    # A two-step global window keeps one completed step between steps, one per global cache.
    global_memories = [memory for memory in cache.global_memories if memory is not None]
    assert len(global_memories) == len(model.global_projections())
    assert all(memory.key.shape[2] == TOKENS_PER_STEP for memory in global_memories)


@pytest.mark.parametrize("kind", KINDS)
def test_cache_vectors_per_token_matches_the_retained_tensors(kind: str) -> None:
    model = tiny(kind)
    inputs = sample(batch=1, steps=4)
    frames = inputs[0]
    actions = inputs[1]
    cache = None
    with torch.no_grad():
        for step in range(actions.shape[1]):
            result = model.forward_step(frames[:, step], actions[:, step], cache)
            cache = result.cache
    if cache is None:
        raise AssertionError("incremental model produced no cache")
    history_tokens = actions.shape[1] * TOKENS_PER_STEP
    global_memories = [memory for memory in cache.global_memories if memory is not None]
    if len(global_memories) != len(model.global_projections()):
        raise AssertionError("model did not fill every global memory")
    layer_memories = [memory for memory in cache.layers if memory is not None]
    if kind in ("per_layer", "ced_full"):
        # Every layer cache grows with the history, so all of them count toward the figure.
        growing = layer_memories + global_memories
    else:
        # The lower caches hold at most one completed step and never grow. Only the global
        # memories count.
        assert all(memory.key.shape[2] <= TOKENS_PER_STEP for memory in layer_memories)
        growing = global_memories
    retained_values = [tensor for memory in growing for tensor in (memory.key, memory.value)]
    retained_scalars = sum(tensor.numel() for tensor in retained_values)
    scalars_per_vector_history = history_tokens * model.settings.d_model
    assert retained_scalars % scalars_per_vector_history == 0
    retained_vectors_per_token = retained_scalars // scalars_per_vector_history
    assert retained_vectors_per_token == model.cache_vectors_per_token()


def test_tile_target_mask_expands_each_step_over_its_cells() -> None:
    step_mask = torch.tensor([[True, False, True]])
    mask = tile_target_mask(step_mask)
    assert mask.shape == (1, 3 * TILES_PER_FRAME)
    assert bool(mask[0, :TILES_PER_FRAME].all())
    assert not bool(mask[0, TILES_PER_FRAME : 2 * TILES_PER_FRAME].any())
    assert int(mask.sum()) == 2 * TILES_PER_FRAME


def test_next_frame_loss_scores_each_step_against_the_following_step() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 3, TILE_VOCAB, TILES_PER_FRAME)
    frames = torch.randint(0, TILE_VOCAB, (1, 3, ROOM_ROWS, ROOM_COLS))
    one_target = torch.zeros(1, 3 * TILES_PER_FRAME, dtype=torch.bool)
    # Mark one target: tile 7 of step 2, predicted from position 7 of step 1.
    one_target[0, 2 * TILES_PER_FRAME + 7] = True
    single = masked_mean(next_frame_losses(logits, frames), one_target)
    expected = torch.nn.functional.cross_entropy(
        logits[0, 1, :, 7][None],
        frames[0, 2].reshape(-1)[7:8],
    )
    torch.testing.assert_close(single, expected)
    with pytest.raises(ValueError, match="no target"):
        masked_mean(next_frame_losses(logits, frames), torch.zeros_like(one_target))
