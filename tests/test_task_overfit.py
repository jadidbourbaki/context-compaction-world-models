from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from shared_cache_world_model.task_overfit import (
    QUERY_STEP,
    SEQUENCE_STEPS,
    SOURCE_STEP,
    TARGET_STEP,
    final_transition_loss,
    shuffle_sources,
    task_batch,
)
from shared_cache_world_model.world import ROOM_COLS, ROOM_ROWS, TILE_VOCAB


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    room_pairs=st.integers(min_value=1, max_value=6),
)
@hypothesis_settings(max_examples=10, deadline=None)
def test_task_requires_distant_room_retrieval(seed: int, room_pairs: int) -> None:
    rng = np.random.default_rng(seed)
    batch = task_batch(room_pairs, rng, torch.device("cpu"))
    frames = batch[0]
    actions = batch[1]
    assert frames.shape == (room_pairs * 2, SEQUENCE_STEPS, ROOM_ROWS, ROOM_COLS)
    assert actions.shape == (room_pairs * 2, SEQUENCE_STEPS)
    assert torch.equal(frames[0::2, 1 : QUERY_STEP + 1], frames[1::2, 1 : QUERY_STEP + 1])
    assert torch.equal(actions[0::2], actions[1::2])
    assert torch.equal(frames[:, SOURCE_STEP], frames[:, TARGET_STEP])
    source_differences = frames[0::2, SOURCE_STEP] != frames[1::2, SOURCE_STEP]
    assert bool(source_differences.flatten(start_dim=1).any(dim=1).all())
    shuffled = shuffle_sources(frames)
    assert torch.equal(shuffled[0::2, SOURCE_STEP], frames[1::2, SOURCE_STEP])
    assert torch.equal(shuffled[1::2, SOURCE_STEP], frames[0::2, SOURCE_STEP])
    assert torch.equal(shuffled[:, 1:], frames[:, 1:])


def test_final_transition_loss_uses_the_query_prediction() -> None:
    batch_size = 2
    positions = ROOM_ROWS * ROOM_COLS
    frames = torch.zeros(batch_size, SEQUENCE_STEPS, ROOM_ROWS, ROOM_COLS, dtype=torch.int64)
    logits = torch.zeros(batch_size, SEQUENCE_STEPS, TILE_VOCAB, positions)
    targets = frames[:, TARGET_STEP].reshape(batch_size, positions).long()
    logits[:, QUERY_STEP].scatter_(1, targets[:, None], 20.0)
    loss = final_transition_loss(logits, frames)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    wrong_query = logits.clone()
    wrong_query[:, QUERY_STEP] = 0.0
    wrong_loss = final_transition_loss(wrong_query, frames)
    assert float(wrong_loss) > float(loss)
