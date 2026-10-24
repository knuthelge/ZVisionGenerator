"""Default-off Sol-Attn/MacSol reference attention for MiniMax-H3 packed rows.

This module is intentionally not wired into generation.  It is a small correctness and
statistics reference for H3-like Q/K/V tensors in SDPA layout ``[B, H, S, D]``.  The
implementation preserves the source-backed Sol-Attn semantics that matter before writing an
MLX/Metal sparse kernel:

* 64-token query/KV blocks by default;
* K summaries are per-block means and V summaries are per-block sums;
* threshold routing is per batch/head/query-block using ``mean + tau * std`` over
  scaled proxy logits and a strict ``score > threshold`` comparison; this is the only active
  routing mode;
* neighboring KV blocks ``abs(q_block - kv_block) <= 1`` are exact;
* a contiguous exact-KV sink is rounded outward to whole blocks;
* non-exact blocks are approximated, not dropped, and are combined with exact blocks in one
  softmax denominator/numerator.

The H3 helper treats the packed layout as
``[text | conditioning video | audio | target video]``: the prefix before target video is the
exact-KV sink, and those prefix query rows are replaced by dense attention outputs so prompt,
conditioning-video, and audio rows remain exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import mlx.core as mx
import numpy as np


BlockRange = tuple[int, int]


@dataclass(frozen=True)
class H3PackedLengths:
    """Token counts for the H3 packed sequence.

    The order is ``[text | conditioning video | audio | target video]``.  The prefix before the
    target-video tail is the default exact-KV sink used by :func:`h3_macsol_reference_attention`.
    """

    text_tokens: int
    conditioning_video_tokens: int
    audio_tokens: int
    target_video_tokens: int

    def __post_init__(self) -> None:
        for name in ("text_tokens", "conditioning_video_tokens", "audio_tokens", "target_video_tokens"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.target_video_tokens <= 0:
            raise ValueError("target_video_tokens must be positive for an H3 packed target tail")

    @property
    def prefix_tokens(self) -> int:
        return int(self.text_tokens + self.conditioning_video_tokens + self.audio_tokens)

    @property
    def sequence_length(self) -> int:
        return int(self.prefix_tokens + self.target_video_tokens)

    def as_dict(self) -> dict[str, int]:
        return {
            "text_tokens": int(self.text_tokens),
            "conditioning_video_tokens": int(self.conditioning_video_tokens),
            "audio_tokens": int(self.audio_tokens),
            "target_video_tokens": int(self.target_video_tokens),
        }


@dataclass(frozen=True)
class MacSolReferenceConfig:
    """Configuration for the default-off source-faithful reference path.

    ``sink_tokens`` names a contiguous exact-KV range starting at ``sink_start``.  The range is
    rounded outward to whole ``block_size`` blocks before routing.  ``prefix_query_tokens`` rows
    are computed densely (or equivalently replaced by dense outputs) to model H3 prefix-query
    exactness without making generation defaults depend on this module.

    Only ``routing_mode='threshold'`` is active.  The previously explored ``budget_topk`` and
    ``threshold_h3_structure``/spatial-tube variants were rejected during local admission sweeps and
    are intentionally not runnable from this reference module.
    """

    block_size: int = 64
    tau: float = 1.0
    sink_start: int = 0
    sink_tokens: int = 0
    prefix_query_tokens: int = 0
    force_prefix_queries_dense: bool = True
    include_neighbors: bool = True
    routing_mode: str = "threshold"

    def validate_for_sequence(self, sequence_length: int) -> None:
        if int(self.block_size) != self.block_size or self.block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {self.block_size!r}")
        routing_mode = str(self.routing_mode)
        if routing_mode != "threshold":
            raise ValueError(
                "only source-faithful threshold routing is active; rejected variants "
                "'budget_topk', 'threshold_h3_structure', and spatial-tube modes are archived provenance only, "
                f"got {self.routing_mode!r}"
            )
        if not math.isfinite(float(self.tau)):
            raise ValueError(f"tau must be finite, got {self.tau!r}")
        if int(self.sink_start) != self.sink_start or self.sink_start < 0:
            raise ValueError(f"sink_start must be a non-negative integer, got {self.sink_start!r}")
        if int(self.sink_tokens) != self.sink_tokens or self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must be a non-negative integer, got {self.sink_tokens!r}")
        if int(self.prefix_query_tokens) != self.prefix_query_tokens or self.prefix_query_tokens < 0:
            raise ValueError(
                f"prefix_query_tokens must be a non-negative integer, got {self.prefix_query_tokens!r}"
            )
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if self.sink_start > sequence_length:
            raise ValueError(f"sink_start {self.sink_start} exceeds sequence length {sequence_length}")
        if self.sink_start + self.sink_tokens > sequence_length:
            raise ValueError(
                f"sink range [{self.sink_start}:{self.sink_start + self.sink_tokens}] exceeds sequence length "
                f"{sequence_length}"
            )
        if self.prefix_query_tokens > sequence_length:
            raise ValueError(
                f"prefix_query_tokens {self.prefix_query_tokens} exceeds sequence length {sequence_length}"
            )


@dataclass(frozen=True)
class MacSolRouting:
    """Block-routing decisions and debug metadata for the reference path."""

    q_block_ranges: tuple[BlockRange, ...]
    kv_block_ranges: tuple[BlockRange, ...]
    threshold_block_mask: np.ndarray
    neighbor_block_mask: np.ndarray
    sink_block_mask: np.ndarray
    exact_block_mask: np.ndarray
    proxy_scores: np.ndarray
    thresholds: np.ndarray
    sink_request: tuple[int, int]
    rounded_sink_token_range: tuple[int, int]
    rounded_sink_block_range: tuple[int, int]
    prefix_query_tokens: int
    block_size: int
    tau: float
    routing_mode: str

    def selected_kv_blocks(self, batch: int, head: int, q_block: int) -> list[int]:
        """Return exact KV block indices for one routed batch/head/query-block row."""
        return np.nonzero(self.exact_block_mask[batch, head, q_block])[0].astype(int).tolist()

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable routing and threshold statistics."""
        total_pairs = int(self.exact_block_mask.size)
        exact_count = int(np.count_nonzero(self.exact_block_mask))
        threshold_count = int(np.count_nonzero(self.threshold_block_mask))
        neighbor_count = int(np.count_nonzero(self.neighbor_block_mask))
        sink_count = int(np.count_nonzero(self.sink_block_mask))
        required_count = int(np.count_nonzero(self.neighbor_block_mask | self.sink_block_mask))
        approximate_count = int(total_pairs - exact_count)
        kv_lengths = [hi - lo for lo, hi in self.kv_block_ranges]
        q_lengths = [hi - lo for lo, hi in self.q_block_ranges]
        exact_per_q = self.exact_block_mask.sum(axis=-1)
        proxy = self.proxy_scores
        thresholds = self.thresholds
        return {
            "block_size": int(self.block_size),
            "routing_mode": str(self.routing_mode),
            "tau": float(self.tau),
            "q_block_count": len(self.q_block_ranges),
            "kv_block_count": len(self.kv_block_ranges),
            "q_block_ranges": [list(item) for item in self.q_block_ranges],
            "kv_block_ranges": [list(item) for item in self.kv_block_ranges],
            "q_block_token_lengths": q_lengths,
            "kv_block_token_lengths": kv_lengths,
            "tail_q_block_tokens": q_lengths[-1],
            "tail_kv_block_tokens": kv_lengths[-1],
            "sink_request": {"start": int(self.sink_request[0]), "tokens": int(self.sink_request[1])},
            "rounded_sink_token_range": list(self.rounded_sink_token_range),
            "rounded_sink_block_range": list(self.rounded_sink_block_range),
            "rounded_sink_block_count": int(self.rounded_sink_block_range[1] - self.rounded_sink_block_range[0]),
            "prefix_query_tokens": int(self.prefix_query_tokens),
            "total_block_pairs": total_pairs,
            "threshold_selected_block_pairs": threshold_count,
            "required_block_pairs_before_primary_selection": required_count,
            "neighbor_exact_block_pairs": neighbor_count,
            "sink_exact_block_pairs": sink_count,
            "exact_block_pairs": exact_count,
            "approximate_block_pairs": approximate_count,
            "exact_block_ratio": exact_count / total_pairs if total_pairs else 0.0,
            "approximate_block_ratio": approximate_count / total_pairs if total_pairs else 0.0,
            "min_exact_blocks_per_query_block": int(exact_per_q.min()) if exact_per_q.size else 0,
            "max_exact_blocks_per_query_block": int(exact_per_q.max()) if exact_per_q.size else 0,
            "mean_exact_blocks_per_query_block": float(exact_per_q.mean()) if exact_per_q.size else 0.0,
            "proxy_score_min": float(proxy.min()) if proxy.size else 0.0,
            "proxy_score_max": float(proxy.max()) if proxy.size else 0.0,
            "threshold_min": float(thresholds.min()) if thresholds.size else 0.0,
            "threshold_max": float(thresholds.max()) if thresholds.size else 0.0,
            "routing_rule": "score > mean + tau * std",
            "strict_threshold_rule": "score > mean + tau * std",
            "non_exact_blocks_are_approximated": True,
        }


@dataclass(frozen=True)
class MacSolAttentionResult:
    """Reference attention output plus routing metadata."""

    output: mx.array
    routing: MacSolRouting
    stats: dict[str, Any]


def block_ranges(sequence_length: int, block_size: int = 64) -> tuple[BlockRange, ...]:
    """Partition a sequence into contiguous token blocks."""
    sequence_length = int(sequence_length)
    block_size = int(block_size)
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return tuple((start, min(start + block_size, sequence_length)) for start in range(0, sequence_length, block_size))


def rounded_sink_blocks(
    sequence_length: int,
    *,
    sink_start: int,
    sink_tokens: int,
    block_size: int = 64,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Round a contiguous sink token range outward to whole blocks.

    Returns ``((token_start, token_stop), (block_start, block_stop))``.  Empty sinks return the
    empty range at ``sink_start`` and a zero-width block range.
    """
    sequence_length = int(sequence_length)
    sink_start = int(sink_start)
    sink_tokens = int(sink_tokens)
    block_size = int(block_size)
    if sequence_length <= 0 or block_size <= 0:
        raise ValueError("sequence_length and block_size must be positive")
    if sink_start < 0 or sink_tokens < 0:
        raise ValueError(f"sink_start/sink_tokens must be non-negative, got {sink_start}, {sink_tokens}")
    if sink_start > sequence_length or sink_start + sink_tokens > sequence_length:
        raise ValueError(
            f"sink range [{sink_start}:{sink_start + sink_tokens}] exceeds sequence length {sequence_length}"
        )
    if sink_tokens == 0:
        block = min(sink_start // block_size, math.ceil(sequence_length / block_size))
        return (sink_start, sink_start), (block, block)
    block_start = sink_start // block_size
    block_stop = math.ceil((sink_start + sink_tokens) / block_size)
    token_start = block_start * block_size
    token_stop = min(sequence_length, block_stop * block_size)
    return (token_start, token_stop), (block_start, block_stop)


def h3_macsol_config(
    lengths: H3PackedLengths,
    *,
    tau: float = 1.0,
    block_size: int = 64,
    routing_mode: str = "threshold",
) -> MacSolReferenceConfig:
    """Return the default H3 threshold-reference config for ``[text | cond-video | audio | target-video]``."""
    if lengths.sequence_length <= 0:
        raise ValueError("H3 packed sequence must be non-empty")
    return MacSolReferenceConfig(
        block_size=int(block_size),
        tau=float(tau),
        sink_start=0,
        sink_tokens=lengths.prefix_tokens,
        prefix_query_tokens=lengths.prefix_tokens,
        force_prefix_queries_dense=True,
        include_neighbors=True,
        routing_mode=str(routing_mode),
    )


def _validate_qkv(q: mx.array, k: mx.array, v: mx.array) -> tuple[int, int, int, int]:
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise ValueError(f"q/k/v must all be rank-4 [B,H,S,D], got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"self-attention reference requires identical q/k/v shapes, got {q.shape}, {k.shape}, {v.shape}")
    batch, heads, sequence, head_dim = (int(dim) for dim in q.shape)
    if batch <= 0 or heads <= 0 or sequence <= 0 or head_dim <= 0:
        raise ValueError(f"q/k/v dimensions must be positive, got {q.shape}")
    return batch, heads, sequence, head_dim


def _stack_block_summaries(x: mx.array, ranges: Iterable[BlockRange], *, reducer: str) -> mx.array:
    summaries: list[mx.array] = []
    x32 = x.astype(mx.float32)
    for lo, hi in ranges:
        block = x32[:, :, lo:hi, :]
        if reducer == "mean":
            summaries.append(mx.mean(block, axis=2))
        elif reducer == "sum":
            summaries.append(mx.sum(block, axis=2))
        else:
            raise ValueError(f"unknown reducer {reducer!r}")
    return mx.stack(summaries, axis=2)


def build_macsol_routing(
    q: mx.array,
    k: mx.array,
    config: MacSolReferenceConfig | None = None,
    *,
    scale: float | None = None,
) -> MacSolRouting:
    """Build source-faithful Sol-Attn threshold block routing for H3 self-attention Q/K tensors.

    The returned masks have shape ``[B, H, q_blocks, kv_blocks]``.  ``threshold_block_mask`` is
    only the strict standardized cutoff decision; ``exact_block_mask`` additionally includes local
    neighbor blocks and the rounded sink blocks.
    """
    batch, heads, sequence, head_dim = _validate_qkv(q, k, k)
    config = config or MacSolReferenceConfig()
    config.validate_for_sequence(sequence)
    block_size = int(config.block_size)
    scale = float(head_dim ** -0.5 if scale is None else scale)

    q_ranges = block_ranges(sequence, block_size)
    kv_ranges = q_ranges
    q_mean = _stack_block_summaries(q, q_ranges, reducer="mean")
    k_mean = _stack_block_summaries(k, kv_ranges, reducer="mean")
    proxy_scores_mx = mx.sum(q_mean[:, :, :, None, :] * k_mean[:, :, None, :, :], axis=-1) * scale
    row_mean = mx.mean(proxy_scores_mx, axis=-1, keepdims=True)
    row_var = mx.mean(mx.square(proxy_scores_mx - row_mean), axis=-1, keepdims=True)
    thresholds_mx = row_mean + float(config.tau) * mx.sqrt(row_var)
    mx.eval(proxy_scores_mx, thresholds_mx)

    proxy_scores = np.asarray(proxy_scores_mx, dtype=np.float32)
    thresholds = np.asarray(thresholds_mx, dtype=np.float32)

    q_blocks = len(q_ranges)
    kv_blocks = len(kv_ranges)
    neighbor_2d = np.zeros((q_blocks, kv_blocks), dtype=bool)
    if config.include_neighbors:
        for qi in range(q_blocks):
            for kj in range(kv_blocks):
                if abs(qi - kj) <= 1:
                    neighbor_2d[qi, kj] = True
    neighbor_mask = np.broadcast_to(neighbor_2d, (batch, heads, q_blocks, kv_blocks)).copy()

    rounded_sink_tokens, rounded_sink_block_range = rounded_sink_blocks(
        sequence,
        sink_start=int(config.sink_start),
        sink_tokens=int(config.sink_tokens),
        block_size=block_size,
    )
    sink_2d = np.zeros((q_blocks, kv_blocks), dtype=bool)
    sink_block_start, sink_block_stop = rounded_sink_block_range
    if sink_block_stop > sink_block_start:
        sink_2d[:, sink_block_start:sink_block_stop] = True
    sink_mask = np.broadcast_to(sink_2d, (batch, heads, q_blocks, kv_blocks)).copy()

    required_mask = neighbor_mask | sink_mask
    threshold_mask = proxy_scores > thresholds  # source-backed strict comparison.
    exact_mask = threshold_mask | required_mask
    return MacSolRouting(
        q_block_ranges=q_ranges,
        kv_block_ranges=kv_ranges,
        threshold_block_mask=threshold_mask,
        neighbor_block_mask=neighbor_mask,
        sink_block_mask=sink_mask,
        exact_block_mask=exact_mask,
        proxy_scores=proxy_scores,
        thresholds=thresholds,
        sink_request=(int(config.sink_start), int(config.sink_tokens)),
        rounded_sink_token_range=rounded_sink_tokens,
        rounded_sink_block_range=rounded_sink_block_range,
        prefix_query_tokens=int(config.prefix_query_tokens),
        block_size=block_size,
        tau=float(config.tau),
        routing_mode=str(config.routing_mode),
    )


def dense_attention_reference(q: mx.array, k: mx.array, v: mx.array, *, scale: float | None = None) -> mx.array:
    """Dense self-attention reference for ``[B,H,S,D]`` tensors."""
    _validate_qkv(q, k, v)
    scale = float(q.shape[-1] ** -0.5 if scale is None else scale)
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    scores = mx.matmul(q32, k32.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, v32)


def _stable_exact_approx_block(
    q_block: mx.array,
    exact_k: mx.array | None,
    exact_v: mx.array | None,
    approx_k: mx.array | None,
    approx_vsum: mx.array | None,
    approx_lengths: mx.array | None,
    *,
    scale: float,
    head_dim: int,
) -> mx.array:
    pieces: list[mx.array] = []
    exact_scores = None
    approx_scores = None
    if exact_k is not None and exact_k.shape[0] > 0:
        exact_scores = mx.matmul(q_block, exact_k.T) * scale
        pieces.append(mx.max(exact_scores, axis=-1, keepdims=True))
    if approx_k is not None and approx_k.shape[0] > 0:
        approx_scores = mx.matmul(q_block, approx_k.T) * scale
        pieces.append(mx.max(approx_scores, axis=-1, keepdims=True))
    if not pieces:
        raise ValueError("each query block must have at least one exact or approximate KV block")
    row_max = pieces[0] if len(pieces) == 1 else mx.maximum(pieces[0], pieces[1])
    for piece in pieces[2:]:
        row_max = mx.maximum(row_max, piece)

    tokens = int(q_block.shape[0])
    numerator = mx.zeros((tokens, head_dim), dtype=mx.float32)
    denominator = mx.zeros((tokens, 1), dtype=mx.float32)
    if exact_scores is not None and exact_v is not None:
        exact_weights = mx.exp(exact_scores - row_max)
        numerator = numerator + mx.matmul(exact_weights, exact_v)
        denominator = denominator + mx.sum(exact_weights, axis=-1, keepdims=True)
    if approx_scores is not None and approx_vsum is not None and approx_lengths is not None:
        approx_weights = mx.exp(approx_scores - row_max)
        numerator = numerator + mx.matmul(approx_weights, approx_vsum)
        denominator = denominator + mx.sum(approx_weights * approx_lengths[None, :], axis=-1, keepdims=True)
    return numerator / denominator


def macsol_reference_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: MacSolReferenceConfig | None = None,
    *,
    scale: float | None = None,
) -> MacSolAttentionResult:
    """Run default-off Sol-Attn reference attention on H3-like SDPA Q/K/V tensors.

    The function returns approximate Sol-Attn outputs for non-prefix query rows.  If
    ``config.force_prefix_queries_dense`` is true, the first ``prefix_query_tokens`` rows are
    overwritten by exact dense attention to model H3's safer prefix-query handling.
    """
    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    config = config or MacSolReferenceConfig()
    config.validate_for_sequence(sequence)
    scale = float(head_dim ** -0.5 if scale is None else scale)
    routing = build_macsol_routing(q, k, config, scale=scale)

    k_mean = _stack_block_summaries(k, routing.kv_block_ranges, reducer="mean")
    v_sum = _stack_block_summaries(v, routing.kv_block_ranges, reducer="sum")
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    kv_lengths = np.array([hi - lo for lo, hi in routing.kv_block_ranges], dtype=np.float32)

    batch_outputs: list[mx.array] = []
    for b in range(batch):
        head_outputs: list[mx.array] = []
        for h in range(heads):
            q_outputs: list[mx.array] = []
            for qi, (qlo, qhi) in enumerate(routing.q_block_ranges):
                exact_blocks = np.nonzero(routing.exact_block_mask[b, h, qi])[0].astype(int).tolist()
                approx_blocks = [idx for idx in range(len(routing.kv_block_ranges)) if idx not in exact_blocks]

                exact_k_parts = []
                exact_v_parts = []
                for kj in exact_blocks:
                    klo, khi = routing.kv_block_ranges[kj]
                    exact_k_parts.append(k32[b, h, klo:khi, :])
                    exact_v_parts.append(v32[b, h, klo:khi, :])
                exact_k = mx.concatenate(exact_k_parts, axis=0) if exact_k_parts else None
                exact_v = mx.concatenate(exact_v_parts, axis=0) if exact_v_parts else None

                if approx_blocks:
                    approx_k = mx.stack([k_mean[b, h, kj, :] for kj in approx_blocks], axis=0)
                    approx_vsum = mx.stack([v_sum[b, h, kj, :] for kj in approx_blocks], axis=0)
                    approx_lengths = mx.array(kv_lengths[approx_blocks], dtype=mx.float32)
                else:
                    approx_k = None
                    approx_vsum = None
                    approx_lengths = None

                q_outputs.append(
                    _stable_exact_approx_block(
                        q32[b, h, qlo:qhi, :],
                        exact_k,
                        exact_v,
                        approx_k,
                        approx_vsum,
                        approx_lengths,
                        scale=scale,
                        head_dim=head_dim,
                    )
                )
            head_outputs.append(mx.concatenate(q_outputs, axis=0))
        batch_outputs.append(mx.stack(head_outputs, axis=0))
    output = mx.stack(batch_outputs, axis=0)

    prefix = int(config.prefix_query_tokens)
    if config.force_prefix_queries_dense and prefix > 0:
        dense = dense_attention_reference(q, k, v, scale=scale)
        if prefix == sequence:
            output = dense
        else:
            output = mx.concatenate([dense[:, :, :prefix, :], output[:, :, prefix:, :]], axis=2)

    stats = routing.summary()
    stats.update(
        {
            "batch": batch,
            "heads": heads,
            "sequence_length": sequence,
            "head_dim": head_dim,
            "scale": scale,
            "prefix_queries_dense": bool(config.force_prefix_queries_dense and prefix > 0),
            "prefix_query_dense_token_count": prefix if config.force_prefix_queries_dense else 0,
            "reference_only_not_generation_path": True,
        }
    )
    return MacSolAttentionResult(output=output, routing=routing, stats=stats)


def h3_macsol_reference_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    lengths: H3PackedLengths,
    *,
    tau: float = 1.0,
    block_size: int = 64,
    routing_mode: str = "threshold",
    scale: float | None = None,
) -> MacSolAttentionResult:
    """Run the H3 packed-layout threshold reference with prefix sink and prefix-query exactness."""
    if int(q.shape[2]) != lengths.sequence_length:
        raise ValueError(
            f"q/k/v sequence length {int(q.shape[2])} does not match H3 lengths {lengths.sequence_length}"
        )
    return macsol_reference_attention(
        q,
        k,
        v,
        h3_macsol_config(
            lengths,
            tau=tau,
            block_size=block_size,
            routing_mode=routing_mode,
        ),
        scale=scale,
    )


__all__ = [
    "H3PackedLengths",
    "MacSolAttentionResult",
    "MacSolReferenceConfig",
    "MacSolRouting",
    "block_ranges",
    "build_macsol_routing",
    "dense_attention_reference",
    "h3_macsol_config",
    "h3_macsol_reference_attention",
    "macsol_reference_attention",
    "rounded_sink_blocks",
]
