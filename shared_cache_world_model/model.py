"""Three world models: a cache in every layer, one shared pair, or one pair per upper layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, model_validator
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from shared_cache_world_model.data import TILES_PER_FRAME
from shared_cache_world_model.world import ACTIONS, TILE_VOCAB

# Each step is one token per tile type followed by one action token, 9 tokens in all.
TOKENS_PER_STEP = TILE_VOCAB + 1
# Actions share the embedding table with tiles and sit after the 8 tile ids.
ACTION_OFFSET = TILE_VOCAB
VOCAB = TILE_VOCAB + len(ACTIONS)
Kind = Literal["per_layer", "shared_cache", "ced", "ced_full"]
# The shared-cache model's own stacks see this many steps: the current one and the one before.
LOCAL_STEPS = 2
# FlexAttention needs torch.compile to run its fused kernel, and its backward pass exists only
# on the GPU. The GPU path uses the compiled kernel. The CPU path, which only ever runs the
# tests and short smoke runs, materializes the mask and calls the plain fused kernel instead.
# Every GPU call uses the same window length, so the kernel is compiled for fixed shapes.
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class ModelSettings(BaseModel):
    kind: Kind = Field(
        description=(
            "per_layer: a KV cache in every layer. shared_cache: one global KV pair read by "
            "every upper layer. ced: one KV pair per upper layer, all projected from the same "
            "windowed lower output. ced_full: ced with ordinary full-attention lower blocks, as "
            "DeepSeek-V4.1-Flash deploys it"
        )
    )
    d_model: int = Field(default=384, ge=8)
    n_heads: int = Field(default=6, ge=1)
    n_layers: int = Field(default=8, ge=2)
    # The longest history, in steps, the model will ever see. Sets the step position table.
    max_history: int = Field(default=128, ge=1)
    per_layer_window: int | None = Field(
        default=None,
        ge=2,
        description="rolling self-attention window for per_layer blocks and ced_full lower blocks",
    )
    global_window: int | None = Field(
        default=None,
        ge=2,
        description="rolling cross-attention window for the shared_cache and ced arms, in steps",
    )

    @model_validator(mode="after")
    def validate_attention_width(self) -> ModelSettings:
        """Require equal, even head widths for multi-head attention and rotary positions."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        head_width = self.d_model // self.n_heads
        if head_width % 2 != 0:
            raise ValueError(f"attention head width {head_width} must be even for rotary positions")
        if self.kind not in ("per_layer", "ced_full") and self.per_layer_window is not None:
            raise ValueError("per_layer_window applies only to the per_layer and ced_full models")
        if self.kind == "per_layer" and self.global_window is not None:
            raise ValueError("global_window applies only to the shared_cache and ced models")
        if self.per_layer_window is not None and self.per_layer_window > self.max_history:
            raise ValueError(
                f"per_layer_window {self.per_layer_window} exceeds max_history {self.max_history}"
            )
        if self.global_window is not None and self.global_window > self.max_history:
            raise ValueError(
                f"global_window {self.global_window} exceeds max_history {self.max_history}"
            )
        return self


@cache
def step_mask(window: int | None) -> MaskMod:
    """The rule for which keys a query may see, by the steps the two tokens belong to."""

    # One function object per window, so the compiled kernel sees the same rule on every call
    # and never recompiles for it.
    def visible(
        batch: torch.Tensor, head: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        # A query at step t sees every token of steps t - window + 1 through t, and of every
        # step up to t when window is None. Tokens inside one step all see each other.
        # FlexAttention calls this with integer index tensors and expects a bool tensor back.
        query_step = query // TOKENS_PER_STEP
        key_step = key // TOKENS_PER_STEP
        not_after = key_step <= query_step
        if window is None:
            return not_after
        return not_after & (query_step - key_step < window)

    return visible


@cache
def block_mask(length: int, window: int | None, device: torch.device) -> BlockMask:
    """The FlexAttention block mask for ``length`` tokens under ``step_mask(window)``."""
    # FlexAttention evaluates the rule per 128 by 128 block and skips blocks with nothing visible,
    # so attention never holds the 1,152 by 1,152 matrix of a 128-step window. Building the
    # block mask evaluates the rule over that matrix once. The cache keeps the result so a run
    # builds each mask one time.
    return create_block_mask(step_mask(window), None, None, length, length, device=device)


@cache
def dense_mask(length: int, window: int | None, device: torch.device) -> torch.Tensor:
    """The (length, length) boolean mask under ``step_mask(window)``, for the CPU path."""
    # The rule ignores the batch and head indices, so a zero stands in for both.
    positions = torch.arange(length, device=device)
    unused_index = torch.zeros((), dtype=torch.int64, device=device)
    return step_mask(window)(unused_index, unused_index, positions[:, None], positions[None, :])


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int | None) -> torch.Tensor:
    """Attention under the step rule: (batch, heads, positions, d_head) in and out."""
    length = q.shape[2]
    if q.device.type == "cuda":
        # FlexAttention returns a tuple only when asked for the log-sum-exp, which the type
        # checker cannot see from the call.
        attended = compiled_flex_attention(q, k, v, block_mask=block_mask(length, window, q.device))
        if not isinstance(attended, torch.Tensor):
            raise TypeError(f"flex_attention returned {type(attended).__name__}")
        return attended
    return F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask(length, window, q.device))


def project_heads(x: torch.Tensor, projection: nn.Linear, n_heads: int) -> torch.Tensor:
    """Project model vectors and split their width into attention heads."""
    # projected: (batch, positions, d_model), before d_model is divided among the heads.
    projected = projection(x)
    batch = projected.shape[0]
    length = projected.shape[1]
    # headed: (batch, heads, positions, d_head), the layout attention consumes.
    headed = projected.view(batch, length, n_heads, -1)
    return headed.transpose(1, 2)


def apply_step_rotary(x: torch.Tensor, start_step: int = 0) -> torch.Tensor:
    """Apply rotary positions by game step to headed queries or keys."""
    head_width = x.shape[-1]
    if head_width % 2 != 0:
        raise ValueError(f"attention head width {head_width} must be even for rotary positions")
    # positions: (sequence positions,) int64, with one rotary position shared by each step.
    positions = torch.arange(x.shape[2], device=x.device) // TOKENS_PER_STEP + start_step
    # frequencies: (half head width,) float32. Each pair rotates at a different frequency.
    frequencies = torch.arange(head_width // 2, device=x.device, dtype=torch.float32)
    frequencies = 1.0 / (10_000.0 ** (frequencies / (head_width // 2)))
    # angles: (sequence positions, half head width) float32.
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos().to(x.dtype)[None, None]
    sine = angles.sin().to(x.dtype)[None, None]
    first = x[..., : head_width // 2]
    second = x[..., head_width // 2 :]
    return torch.cat([first * cosine - second * sine, first * sine + second * cosine], dim=-1)


@dataclass(frozen=True)
class ProjectedMemory:
    """One key and value pair that upper blocks read through cross-attention."""

    # Both tensors are (batch, heads, positions, d_head).
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class AttentionStep:
    """One attention result and the cache retained for the following step."""

    output: torch.Tensor
    cache: ProjectedMemory


@dataclass(frozen=True)
class BlockStep:
    """One block result and its self-attention cache."""

    output: torch.Tensor
    cache: ProjectedMemory | None


@dataclass(frozen=True)
class ModelCache:
    """The state an incremental model retains between game steps."""

    step: int
    layers: list[ProjectedMemory | None]
    # One entry per distinct global cache: none for per_layer, one for shared_cache, and one per
    # upper block for ced.
    global_memories: list[ProjectedMemory | None]


@dataclass(frozen=True)
class ModelStep:
    """The logits for one game step and the state retained after it."""

    logits: torch.Tensor
    cache: ModelCache


def append_memory(
    previous: ProjectedMemory | None,
    current: ProjectedMemory,
    retained_steps: int | None,
) -> ProjectedMemory:
    """Append one step of K and V, then retain the requested completed steps."""
    if previous is None:
        joined = current
    else:
        # key and value: (batch, heads, cached positions + 9, d_head).
        key = torch.cat([previous.key, current.key], dim=2)
        value = torch.cat([previous.value, current.value], dim=2)
        joined = ProjectedMemory(key=key, value=value)
    if retained_steps is None:
        return joined
    retained_tokens = retained_steps * TOKENS_PER_STEP
    return ProjectedMemory(
        key=joined.key[:, :, -retained_tokens:],
        value=joined.value[:, :, -retained_tokens:],
    )


def attend_to_available(q: torch.Tensor, memory: ProjectedMemory) -> torch.Tensor:
    """Attend one current step to a cache that contains no future positions."""
    if q.device.type == "cuda":
        attended = compiled_flex_attention(q, memory.key, memory.value)
        if not isinstance(attended, torch.Tensor):
            raise TypeError(f"flex_attention returned {type(attended).__name__}")
        return attended
    return F.scaled_dot_product_attention(q, memory.key, memory.value)


class Attention(nn.Module):
    """Multi-head attention that is causal over steps and unrestricted inside a step."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, window: int | None) -> torch.Tensor:
        # x: (batch, steps * 9, d_model), the source of the queries. memory: the same shape,
        # the source of the keys and values. Self-attention passes the same tensor for both.
        # Project, then split d_model into heads: (batch, length, d_model) becomes
        # (batch, heads, length, d_head), so each head attends over its own slice of the width.
        q = project_heads(x, self.query, self.n_heads)
        k = project_heads(memory, self.key, self.n_heads)
        v = project_heads(memory, self.value, self.n_heads)
        q = apply_step_rotary(q)
        k = apply_step_rotary(k)
        # attended: (batch, heads, length, d_head), each query's weighted sum of visible values.
        attended = attend(q, k, v, window)
        # Put the heads side by side again: (batch, length, d_model).
        merged = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        # One linear layer mixes the heads.
        return self.out(merged)

    def forward_step(
        self,
        x: torch.Tensor,
        previous: ProjectedMemory | None,
        retained_steps: int | None,
        step: int,
    ) -> AttentionStep:
        """Attend one complete game step and update this layer's inference cache."""
        # q, key, and value: (batch, heads, 9, d_head), one complete current step.
        q = project_heads(x, self.query, self.n_heads)
        key = project_heads(x, self.key, self.n_heads)
        value = project_heads(x, self.value, self.n_heads)
        q = apply_step_rotary(q, start_step=step)
        key = apply_step_rotary(key, start_step=step)
        current = ProjectedMemory(key=key, value=value)
        available = append_memory(previous, current, retained_steps=None)
        attended = attend_to_available(q, available)
        # merged: (batch, 9, d_model), with the head slices side by side again.
        merged = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        output = self.out(merged)
        next_cache = append_memory(previous, current, retained_steps=retained_steps)
        return AttentionStep(output=output, cache=next_cache)


class SharedKV(nn.Module):
    """Project the normalized lower-stack output into one global K and V pair."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, memory: torch.Tensor, start_step: int = 0) -> ProjectedMemory:
        # key and value: (batch, heads, positions, d_head), projected once from the lower output.
        key = project_heads(memory, self.key, self.n_heads)
        value = project_heads(memory, self.value, self.n_heads)
        key = apply_step_rotary(key, start_step=start_step)
        return ProjectedMemory(key=key, value=value)


class CrossAttention(nn.Module):
    """Attention with layer-specific queries over an already projected global KV pair."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, memory: ProjectedMemory, window: int | None) -> torch.Tensor:
        # q: (batch, heads, positions, d_head), unique to this upper block.
        q = project_heads(x, self.query, self.n_heads)
        q = apply_step_rotary(q)
        # attended: (batch, heads, positions, d_head), read from the projected global K and V
        # under the same step rule as self-attention, bounded to ``window`` steps when set.
        attended = attend(q, memory.key, memory.value, window=window)
        # merged: (batch, positions, d_model), with the head slices side by side again.
        merged = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out(merged)

    def forward_step(self, x: torch.Tensor, memory: ProjectedMemory, step: int) -> torch.Tensor:
        """Attend one complete game step to the available shared global KV."""
        # q: (batch, heads, 9, d_head), unique to this upper block.
        q = project_heads(x, self.query, self.n_heads)
        q = apply_step_rotary(q, start_step=step)
        attended = attend_to_available(q, memory)
        # merged: (batch, 9, d_model), with the head slices side by side again.
        merged = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out(merged)


class Block(nn.Module):
    """Pre-norm transformer block: self-attention, optional cross-attention, feed-forward."""

    def __init__(self, d_model: int, n_heads: int, cross: bool) -> None:
        super().__init__()
        # Each sublayer has its own LayerNorm, applied to its input before it runs.
        self.norm_self = nn.LayerNorm(d_model) if not cross else None
        self.self_attention = Attention(d_model, n_heads) if not cross else None
        # The shared-cache model's upper blocks use cross-attention as their token mixer.
        self.norm_cross = nn.LayerNorm(d_model) if cross else None
        self.cross_attention = CrossAttention(d_model, n_heads) if cross else None
        self.norm_ff = nn.LayerNorm(d_model)
        # The feed-forward layer acts on each token alone: d_model to 4 * d_model and back.
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        window: int | None,
        memory: ProjectedMemory | None = None,
        cross_window: int | None = None,
    ) -> torch.Tensor:
        # x: (batch, steps * 9, d_model) throughout. Every sublayer adds to x, the residual.
        if self.self_attention is not None and self.norm_self is not None:
            normed = self.norm_self(x)
            x = x + self.self_attention(normed, normed, window)
        if self.cross_attention is not None and self.norm_cross is not None:
            if memory is None:
                raise ValueError("a cross-attention block needs a memory")
            # Queries come from this block. The memory is a projected global KV, the same pair
            # for every upper block in the shared-cache model and this block's own pair in ced.
            x = x + self.cross_attention(self.norm_cross(x), memory, cross_window)
        return x + self.feed_forward(self.norm_ff(x))

    def forward_step(
        self,
        x: torch.Tensor,
        previous: ProjectedMemory | None,
        retained_steps: int | None,
        step: int,
        memory: ProjectedMemory | None = None,
    ) -> BlockStep:
        """Run one game step through a block and update its local or global cache."""
        next_cache = previous
        if self.self_attention is not None and self.norm_self is not None:
            normed = self.norm_self(x)
            self_result = self.self_attention.forward_step(normed, previous, retained_steps, step)
            x = x + self_result.output
            next_cache = self_result.cache
        if self.cross_attention is not None and self.norm_cross is not None:
            if memory is None:
                raise ValueError("a cross-attention block needs a memory")
            cross_output = self.cross_attention.forward_step(self.norm_cross(x), memory, step)
            x = x + cross_output
        output = x + self.feed_forward(self.norm_ff(x))
        return BlockStep(output=output, cache=next_cache)


class WorldModel(nn.Module):
    """Predict the next frame from the steps so far, holding history in one of three ways."""

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.settings = settings
        # A lookup table for the 8 tile types and 4 actions.
        self.token_embedding = nn.Embedding(VOCAB, settings.d_model)
        # Each tile-type token carries a 256-cell occupancy mask projected to model width.
        self.mask_projection = nn.Linear(TILES_PER_FRAME, settings.d_model, bias=False)
        # Two learned position tables, both scaled to 0.02 at initialization like the embeddings.
        # within_step: (1, 1, 9, d_model), one vector per position inside a step.
        # step_position: (1, max_history, 1, d_model), one vector per step of history. The sum
        # places a token. The same tile position of two steps shares the first and differs in
        # the second.
        self.within_step = nn.Parameter(torch.randn(1, 1, TOKENS_PER_STEP, settings.d_model) * 0.02)
        self.step_position = nn.Parameter(
            torch.randn(1, settings.max_history, 1, settings.d_model) * 0.02
        )
        # The two cross-attention models split their layers in two stacks. 8 layers give 4 and 4.
        half = settings.n_layers // 2
        if settings.kind == "per_layer":
            self.lower = nn.ModuleList(
                [
                    Block(settings.d_model, settings.n_heads, cross=False)
                    for _ in range(settings.n_layers)
                ]
            )
            self.upper = nn.ModuleList()
            self.shared_norm = None
            self.shared_kv = None
            self.upper_kv = nn.ModuleList()
        else:
            self.lower = nn.ModuleList(
                [Block(settings.d_model, settings.n_heads, cross=False) for _ in range(half)]
            )
            self.upper = nn.ModuleList(
                [
                    Block(settings.d_model, settings.n_heads, cross=True)
                    for _ in range(settings.n_layers - half)
                ]
            )
            self.shared_norm = nn.LayerNorm(settings.d_model)
            if settings.kind == "shared_cache":
                # One projection, read by every upper block.
                self.shared_kv = SharedKV(settings.d_model, settings.n_heads)
                self.upper_kv = nn.ModuleList()
            else:
                # One projection per upper block, all reading the same normalized lower output.
                self.shared_kv = None
                self.upper_kv = nn.ModuleList(
                    [SharedKV(settings.d_model, settings.n_heads) for _ in range(len(self.upper))]
                )
        self.norm_out = nn.LayerNorm(settings.d_model)

    def global_projections(self) -> list[SharedKV]:
        """The key and value projections the upper blocks read, one per distinct global cache."""
        if self.settings.kind == "shared_cache":
            if self.shared_kv is None:
                raise ValueError("a shared-cache model needs a global KV projection")
            return [self.shared_kv]
        if self.settings.kind in ("ced", "ced_full"):
            return [cast(SharedKV, projection) for projection in self.upper_kv]
        return []

    def memory_index(self, upper_index: int) -> int:
        """Which global cache upper block ``upper_index`` reads."""
        # The shared-cache model has one cache for every upper block. The ced models have one each.
        if self.settings.kind == "shared_cache":
            return 0
        return upper_index

    def lower_window(self) -> int | None:
        """The self-attention window of the lower stack in steps, or None for the full history."""
        # ced_full keeps ordinary lower blocks, so its lower caches grow with the history the way
        # the per-layer model's do, up to the same optional window. The other cross-attention arms
        # bound the lower stack to two steps so that only the global cache grows.
        if self.settings.kind == "ced_full":
            return self.settings.per_layer_window
        return LOCAL_STEPS

    def decode(self, tile_states: torch.Tensor) -> torch.Tensor:
        """Turn tile-type states into one occupancy score per spatial cell."""
        # The transpose ties the decoder to the spatial basis used by the input mask projection.
        return F.linear(tile_states, self.mask_projection.weight.T)

    def input_tokens(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode every tile type's occupancy mask and the action for each step."""
        batch = frames.shape[0]
        steps = frames.shape[1]
        # masks: (batch, steps, 8, 256) float32. Each plane marks one tile type's cells.
        masks = F.one_hot(frames.long(), num_classes=TILE_VOCAB)
        masks = masks.permute(0, 1, 4, 2, 3)
        masks = masks.reshape(batch, steps, TILE_VOCAB, TILES_PER_FRAME).float()
        # tile_ids: (8,) int64. The same eight learned identities occur in every step.
        tile_ids = torch.arange(TILE_VOCAB, device=frames.device)
        # tile_types: (1, 1, 8, d_model), broadcast across batches and steps.
        tile_types = self.token_embedding(tile_ids)[None, None]
        # tile_tokens: (batch, steps, 8, d_model), spatial masks plus tile identities.
        tile_tokens = self.mask_projection(masks) + tile_types
        # action_ids: (batch, steps) int64, shifted beyond the eight tile identities.
        action_ids = actions + ACTION_OFFSET
        action_vectors = self.token_embedding(action_ids)
        # action_tokens: (batch, steps, 1, d_model), one learned vector for the chosen action.
        action_tokens = action_vectors[:, :, None]
        return torch.cat([tile_tokens, action_tokens], dim=2)

    def embed(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Tile-mask and action vectors with positions: (batch, steps * 9, d_model)."""
        batch = frames.shape[0]
        steps = frames.shape[1]
        if steps > self.settings.max_history:
            raise ValueError(f"{steps} steps exceed max_history {self.settings.max_history}")
        # embedded: (batch, steps, 9, d_model), eight tile masks and one action per step.
        embedded = self.input_tokens(frames, actions)
        # Broadcasting adds within_step along the 9-token axis and the first ``steps`` rows of
        # step_position along the steps axis. placed: (batch, steps, 9, d_model).
        placed = embedded + self.within_step + self.step_position[:, :steps]
        # Flatten to the one sequence the attention layers take.
        return placed.reshape(batch, steps * TOKENS_PER_STEP, -1)

    def embed_step(self, frames: torch.Tensor, actions: torch.Tensor, step: int) -> torch.Tensor:
        """Embed one game step at its absolute position in the history."""
        if step >= self.settings.max_history:
            raise ValueError(f"step {step} exceeds max_history {self.settings.max_history}")
        # embedded: (batch, 9, d_model), eight tile masks and one action.
        embedded = self.input_tokens(frames[:, None], actions[:, None])[:, 0]
        within_position = self.within_step[:, 0]
        step_position = self.step_position[:, step]
        return embedded + within_position + step_position

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-tile occupancy logits from each step for the following frame."""
        hidden = self.forward_hidden(frames, actions)
        # tile_states: (batch, steps, 8, d_model), with the action token removed.
        tile_states = hidden[:, :, :TILE_VOCAB]
        # logits: (batch, steps, 8, 256), one occupancy score per type and cell.
        return self.decode(tile_states)

    def forward_last(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Logits for the last step only, the prediction a harness reads after prefill."""
        if self.settings.kind == "per_layer":
            return self.forward(frames, actions)[:, -1]
        steps = actions.shape[1]
        x = self.embed(frames, actions)
        for block in self.lower:
            x = block(x, window=self.lower_window())
        if self.shared_norm is None:
            raise ValueError("a cross-attention model needs a shared normalization")
        source = self.shared_norm(x)
        memories = [projection(source) for projection in self.global_projections()]
        if self.settings.global_window is not None:
            window_tokens = self.settings.global_window * TOKENS_PER_STEP
            memories = [
                ProjectedMemory(
                    key=memory.key[:, :, -window_tokens:],
                    value=memory.value[:, :, -window_tokens:],
                )
                for memory in memories
            ]
        # Upper blocks see earlier steps only through the projected keys and values, so they run
        # on the last step alone. That is the prefill saving. last: (batch, 9, d_model).
        last = x[:, -TOKENS_PER_STEP:]
        for upper_index in range(len(self.upper)):
            block = cast(Block, self.upper[upper_index])
            result = block.forward_step(
                last,
                None,
                retained_steps=None,
                step=steps - 1,
                memory=memories[self.memory_index(upper_index)],
            )
            last = result.output
        tile_states = self.norm_out(last)[:, :TILE_VOCAB]
        # logits: (batch, 8, 256).
        return self.decode(tile_states)

    def forward_hidden(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the final normalized state of every tile-type and action token."""
        # frames: (batch, steps, 16, 16) int64 tile ids. actions: (batch, steps) int64.
        steps = actions.shape[1]
        # x: (batch, steps * 9, d_model), eight tile-mask tokens and one action per step.
        x = self.embed(frames, actions)
        if self.settings.kind == "per_layer":
            # Every layer keeps its own keys and values under the configured history window.
            for block in self.lower:
                x = block(x, window=self.settings.per_layer_window)
        else:
            # The lower stack sees its window, two steps for the windowed arms and the whole
            # history for ced_full. Its normalized output is the one source every global K and V
            # is projected from. The upper blocks then read the history through
            # cross-attention, bounded to global_window steps when set.
            for block in self.lower:
                x = block(x, window=self.lower_window())
            if self.shared_norm is None:
                raise ValueError("a cross-attention model needs a shared normalization")
            source = self.shared_norm(x)
            # memories: one projected pair per distinct global cache, all from the same source.
            memories = [projection(source) for projection in self.global_projections()]
            for upper_index in range(len(self.upper)):
                memory = memories[self.memory_index(upper_index)]
                x = self.upper[upper_index](
                    x, window=LOCAL_STEPS, memory=memory, cross_window=self.settings.global_window
                )
        # hidden: (batch, steps, 9, d_model), including the action token in each step.
        return self.norm_out(x).reshape(frames.shape[0], steps, TOKENS_PER_STEP, -1)

    def forward_step(
        self, frames: torch.Tensor, actions: torch.Tensor, cache: ModelCache | None = None
    ) -> ModelStep:
        """Predict from one game step and return the cache for the next step."""
        layer_count = len(self.lower) + len(self.upper)
        projections = self.global_projections()
        if cache is None:
            cache = ModelCache(
                step=0, layers=[None] * layer_count, global_memories=[None] * len(projections)
            )
        if len(cache.layers) != layer_count:
            raise ValueError(f"cache has {len(cache.layers)} layers, expected {layer_count}")
        if len(cache.global_memories) != len(projections):
            raise ValueError(
                f"cache has {len(cache.global_memories)} global memories, "
                f"expected {len(projections)}"
            )
        # x: (batch, 9, d_model), eight tile-mask tokens and one action.
        x = self.embed_step(frames, actions, cache.step)
        next_layers: list[ProjectedMemory | None] = []
        next_global: list[ProjectedMemory | None] = []
        if self.settings.kind == "per_layer":
            retained_steps = None
            if self.settings.per_layer_window is not None:
                retained_steps = self.settings.per_layer_window - 1
            for index in range(len(self.lower)):
                block = cast(Block, self.lower[index])
                result = block.forward_step(
                    x,
                    cache.layers[index],
                    retained_steps=retained_steps,
                    step=cache.step,
                )
                x = result.output
                next_layers.append(result.cache)
        else:
            # A windowed lower block keeps one step fewer than its window between steps, so the
            # current step completes the window. A full lower block keeps everything.
            lower_window_steps = self.lower_window()
            lower_retained = None if lower_window_steps is None else lower_window_steps - 1
            for index in range(len(self.lower)):
                block = cast(Block, self.lower[index])
                result = block.forward_step(
                    x,
                    cache.layers[index],
                    retained_steps=lower_retained,
                    step=cache.step,
                )
                x = result.output
                next_layers.append(result.cache)
            if self.shared_norm is None:
                raise ValueError("a cross-attention model needs a shared normalization")
            source = self.shared_norm(x)
            # With a global window of w steps, this step attends over w steps and the cache keeps
            # w - 1 of them, so the next step's own K and V complete its window. The rule mirrors
            # the per-layer arm's rolling self-attention cache.
            retained_global = None
            if self.settings.global_window is not None:
                retained_global = self.settings.global_window - 1
            # available: what each upper block attends to at this step.
            available: list[ProjectedMemory] = []
            for index in range(len(projections)):
                current = projections[index](source, start_step=cache.step)
                previous = cache.global_memories[index]
                available.append(append_memory(previous, current, retained_steps=None))
                next_global.append(append_memory(previous, current, retained_steps=retained_global))
            upper_offset = len(self.lower)
            for upper_index in range(len(self.upper)):
                block = cast(Block, self.upper[upper_index])
                layer_index = upper_offset + upper_index
                result = block.forward_step(
                    x,
                    cache.layers[layer_index],
                    retained_steps=LOCAL_STEPS - 1,
                    step=cache.step,
                    memory=available[self.memory_index(upper_index)],
                )
                x = result.output
                next_layers.append(result.cache)
        # logits: (batch, 1, 8, 256), the frame prediction produced by this step.
        tile_states = self.norm_out(x)[:, :TILE_VOCAB]
        logits = self.decode(tile_states)[:, None]
        next_cache = ModelCache(
            step=cache.step + 1,
            layers=next_layers,
            global_memories=next_global,
        )
        return ModelStep(logits=logits, cache=next_cache)

    def cache_vectors_per_token(self) -> int:
        """How many d_model vectors of history the model keeps per token at inference."""
        if self.settings.kind in ("per_layer", "ced_full"):
            # Keys and values in every layer, so 8 layers give 16 vectors. For ced_full the lower
            # blocks keep their own pairs and every upper block keeps its own pair projected from
            # the lower output, so the count matches the per-layer model and the two differ only
            # in the source.
            return 2 * self.settings.n_layers
        if self.settings.kind == "ced":
            # One key and value pair per upper layer, each projected from the same lower output.
            # 8 layers give 4 upper layers and 8 vectors.
            return 2 * len(self.upper)
        # One global key and value pair is reused by every upper layer. The local windows hold
        # two steps and do not grow with history.
        return 2

    def local_receptive_steps(self) -> int:
        """Steps reachable through the shared arm's bounded self-decoder stack."""
        lower_layers = self.settings.n_layers // 2
        return 1 + lower_layers * (LOCAL_STEPS - 1)


def next_frame_losses(logits: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Per-cell cross-entropy of each step's prediction of the following frame."""
    # logits: (batch, steps, 8, 256). Predictions drop the last step and put the eight tile
    # types last for cross_entropy: (batch * (steps - 1) * 256, 8).
    batch = logits.shape[0]
    steps = logits.shape[1]
    predictions = logits[:, :-1].permute(0, 1, 3, 2).reshape(-1, TILE_VOCAB)
    # targets: (batch * (steps - 1) * 256,) int64, every cell of each following frame.
    targets = frames[:, 1:].reshape(-1).long()
    # reduction="none" keeps one loss per target so callers can average under their own masks.
    losses = F.cross_entropy(predictions, targets, reduction="none")
    return losses.view(batch, (steps - 1) * TILES_PER_FRAME)


def shift_to_targets(token_mask: torch.Tensor) -> torch.Tensor:
    """Drop the first frame of a per-cell mask so it lines up with next-frame losses."""
    return token_mask[:, TILES_PER_FRAME:]


def masked_mean(losses: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """The mean of ``losses`` over the targets ``target_mask`` marks. Raises when none is."""
    mask = shift_to_targets(target_mask)
    if not bool(mask.any()):
        raise ValueError("no target in this batch is marked by the mask")
    return losses[mask].mean()


def tile_target_mask(step_mask: torch.Tensor) -> torch.Tensor:
    """Expand a per-step mask over every cell of the corresponding frames."""
    batch = step_mask.shape[0]
    steps = step_mask.shape[1]
    # per_cell: (batch, steps, 256) bool, one copy of each step flag per spatial cell.
    per_cell = step_mask[:, :, None].expand(batch, steps, TILES_PER_FRAME)
    return per_cell.reshape(batch, steps * TILES_PER_FRAME)
