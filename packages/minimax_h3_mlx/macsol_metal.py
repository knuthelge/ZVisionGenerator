"""Default-off MLX/Metal Sol-Attn (MacSol) probes for MiniMax-H3.

This module is intentionally not wired into generation and must not be presented as a production
or generation speed path.  It implements source-faithful threshold-routing MacSol/Sol-Attn
forward probes with MLX custom Metal kernels so tiled/threadgroup-cooperative work can be judged
against :mod:`minimax_h3_mlx.macsol_reference`:

* 64-token (or caller-supplied) Q/K/V blocks;
* Q/K block means and V block sums prepared by a Metal summary kernel;
* per ``[batch, head, query-block]`` threshold ``mean + tau * std`` over scaled centroid logits;
* exact KV blocks for strict threshold hits, local neighbour blocks, and the rounded exact-KV sink;
* dense/exact prefix query rows when requested by the H3 packed-layout policy;
* one sparse online-softmax state per query row, with exact token blocks and centroid/V-sum
  approximation for unselected blocks.

The normal Metal path does not materialize dense ``S x S`` scores, a dense mask, or a persistent
routing tensor.  It returns a contiguous head/query-block subset for internal captured-real-QKV
probes only.  The measured one-thread-per-query-row forward, SIMD-group cooperative v2 forward,
q-block route-sharing v3 forward, q-block/SIMD-group cooperative v4 forward, and exact-tile v5
microkernel are rejected archive/provenance-only paths.  They remain runnable only for bounded
parity/provenance probes and must not be offered by generation or profile surfaces as deployable
speed paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import mlx.core as mx

from .macsol_reference import MacSolReferenceConfig, rounded_sink_blocks


MACSOL_METAL_STATUS = "rejected_scalar_row_thread_provenance_only"
MACSOL_METAL_REJECTION_REASON = (
    "The first source-faithful row-owned MLX/Metal forward matched the reference, but measured "
    "slower than sampled dense and MLX reference baselines; future work needs a materially "
    "different tiling/fusion strategy before any production/generation integration."
)
MACSOL_METAL_V2_STATUS = "rejected_simdgroup_cooperative_provenance_only"
MACSOL_METAL_V2_DESCRIPTION = (
    "A second MacSol forward probe maps four query rows to one 128-thread Metal threadgroup, with "
    "one SIMD-group per row. Lanes cooperatively reduce Q·K scores across the head dimension and "
    "accumulate strided output dimensions from a two-pass stable exact/centroid softmax. It preserves "
    "the same threshold, sink, neighbor, and prefix-query semantics as the scalar provenance kernel "
    "and remains rejected/provenance-only."
)
MACSOL_METAL_V2_REJECTION_REASON = (
    "The simdgroup-cooperative v2 forward is parity-clean and 1.72x faster than the rejected scalar "
    "row-thread probe on the real 960x544 block-10/head-0/middle-q-block subset, but it remains "
    "slower than sampled dense (0.0389s vs 0.0128s) and slightly slower than the MLX subset "
    "reference (0.0389s vs 0.0366s), so it is provenance-only rather than an expansion candidate."
)
MACSOL_METAL_V3_STATUS = "rejected_qblock_tiled_provenance_only"
MACSOL_METAL_V3_DESCRIPTION = (
    "A third default-off MacSol forward probe maps one full query block/head tile to one Metal "
    "threadgroup. Thread 0 computes the threshold/sink/neighbor exact-KV block route once into "
    "threadgroup memory, all 64 query-row threads reuse that route, and each row then performs the "
    "same source-faithful exact-token plus centroid/V-sum approximate online softmax."
)
MACSOL_METAL_V3_REJECTION_REASON = (
    "The q-block tiled v3 forward is parity-clean but not a speed candidate on the real "
    "960x544 block-10/head-0/middle-q-block subset: route reuse leaves scalar per-row exact-token "
    "dot/output work dominant, so it is slower than sampled dense and rejected v2."
)
MACSOL_METAL_V4_STATUS = "rejected_qblock_simdgroup_cooperative_provenance_only"
MACSOL_METAL_V4_DESCRIPTION = (
    "A fourth rejected/provenance-only MacSol forward probe fuses the useful parts of v2 and v3: one Metal "
    "threadgroup owns a full 64-token query block/head tile, computes the source-faithful exact-KV "
    "route once into threadgroup memory, then eight SIMD-groups sweep the 64 query rows in waves. "
    "Each SIMD-group cooperatively reduces exact-token QK scores across head dimension 128 and "
    "accumulates four strided output dimensions per lane while preserving centroid/V-sum "
    "approximation for unselected blocks."
)
MACSOL_METAL_V4_REJECTION_REASON = (
    "The q-block/SIMD-group cooperative v4 forward is parity-clean but not a speed candidate on "
    "the real 960x544 block-10/head-0/middle-q-block subset: median 0.1551s vs sampled dense "
    "0.0129s, MLX subset reference 0.0379s, rejected v2 0.0390s, and rejected v3 0.0668s. "
    "Serially sweeping eight SIMD-group row waves inside one threadgroup reduces launch/route "
    "duplication but under-occupies the GPU and is slower than the prior row-grouped v2 path."
)
MACSOL_METAL_V5_STATUS = "rejected_exact_tile_microkernel_provenance_only"
MACSOL_METAL_V5_DESCRIPTION = (
    "A fifth bounded MacSol prototype is only a 64-query x 64-key exact-attention tile microkernel. "
    "One 256-thread Metal threadgroup computes one or more full BF16 exact tiles, stores only a "
    "transient 64x64 score tile plus row softmax state in threadgroup memory, and emits per-tile "
    "outputs for timing/parity probes. It is not a full sparse online-softmax forward and is not "
    "wired into generation."
)
MACSOL_METAL_V5_REJECTION_REASON = (
    "The v5 exact-tile microkernel is parity-clean, but on the captured real 960x544 block-10/head-0/"
    "q-block-157 subset its amortized exact-tile lower bound for the 166 route-selected exact KV "
    "blocks is 0.0292s versus 0.0151s for sampled dense row-block attention, before adding routing, "
    "approximate blocks, or cross-tile online-softmax combination. It remains provenance-only."
)
MACSOL_METAL_V5_DECISION_NOTE = (
    "Use v5 only as a rejected lower-bound signal for future block-tiled sparse-forward designs; do "
    "not expand it unless a materially different tile/online-softmax architecture changes the measured "
    "exact-block cost model."
)


@dataclass(frozen=True)
class MacSolMetalPreparation:
    """Summary and threshold tensors produced by the Metal preparation path."""

    q_mean: mx.array
    k_mean: mx.array
    v_sum: mx.array
    thresholds: mx.array
    proxy_mean: mx.array
    proxy_std: mx.array
    block_count: int
    block_size: int
    scale: float
    tau: float


@dataclass(frozen=True)
class MacSolMetalAttentionResult:
    """Subset output plus preparation metadata for the rejected internal Metal probe."""

    output: mx.array
    preparation: MacSolMetalPreparation
    head_start: int
    head_count: int
    q_block_start: int
    q_block_count: int
    q_token_range: tuple[int, int]
    rounded_sink_block_range: tuple[int, int]
    prefix_query_tokens: int


@dataclass(frozen=True)
class MacSolMetalExactTileResult:
    """Output and metadata from the v5 exact 64x64 tile microkernel probe."""

    output: mx.array
    head_start: int
    head_count: int
    q_block_start: int
    kv_block_start: int
    kv_block_count: int
    q_token_range: tuple[int, int]
    kv_token_range: tuple[int, int]
    block_size: int
    scale: float


_EPS_DEN = 1_000_000_000


def _ratio_template(value: float, *, denominator: int = _EPS_DEN) -> tuple[int, int]:
    if not math.isfinite(float(value)):
        raise ValueError(f"Metal template scalar must be finite, got {value!r}")
    return int(round(float(value) * denominator)), int(denominator)


@lru_cache(maxsize=1)
def _summary_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_block_summaries",
        input_names=["q", "k", "v"],
        output_names=["q_mean", "k_mean", "v_sum"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint d = elem % D;
            uint block = (elem / D) % NB;
            uint h = (elem / (D * NB)) % H;
            uint b = elem / (D * NB * H);
            uint start = block * uint(BLOCK);
            uint stop = metal::min(start + uint(BLOCK), uint(S));
            uint len = stop - start;

            float q_acc = 0.0f;
            float k_acc = 0.0f;
            float v_acc = 0.0f;
            for (uint s = start; s < stop; ++s) {
                uint idx = (((b * H + h) * S + s) * D + d);
                q_acc += static_cast<float>(q[idx]);
                k_acc += static_cast<float>(k[idx]);
                v_acc += static_cast<float>(v[idx]);
            }
            uint out_idx = (((b * H + h) * NB + block) * D + d);
            float inv_len = 1.0f / static_cast<float>(len);
            q_mean[out_idx] = q_acc * inv_len;
            k_mean[out_idx] = k_acc * inv_len;
            v_sum[out_idx] = v_acc;
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _threshold_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_thresholds",
        input_names=["q_mean", "k_mean"],
        output_names=["thresholds", "proxy_mean", "proxy_std"],
        source=r"""
            uint row = thread_position_in_grid.x;
            uint qb = row % NB;
            uint h = (row / NB) % H;
            uint b = row / (NB * H);
            float scale = static_cast<float>(SCALE_NUM) / static_cast<float>(SCALE_DEN);
            float tau = static_cast<float>(TAU_NUM) / static_cast<float>(TAU_DEN);
            uint q_base = (((b * H + h) * NB + qb) * D);

            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (uint kb = 0; kb < NB; ++kb) {
                uint k_base = (((b * H + h) * NB + kb) * D);
                float dot = 0.0f;
                for (uint d = 0; d < D; ++d) {
                    dot += q_mean[q_base + d] * k_mean[k_base + d];
                }
                float score = dot * scale;
                sum += score;
                sum_sq += score * score;
            }
            float n = static_cast<float>(NB);
            float mean = sum / n;
            float var = metal::max(sum_sq / n - mean * mean, 0.0f);
            float std = metal::sqrt(var);
            thresholds[row] = mean + tau * std;
            proxy_mean[row] = mean;
            proxy_std[row] = std;
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _forward_subset_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_sparse_online_subset",
        input_names=["q", "k", "v", "q_mean", "k_mean", "v_sum", "thresholds"],
        output_names=["out"],
        source=r"""
            uint row = thread_position_in_grid.x;
            uint token_offset = row % TOKENS_OUT;
            uint local_h = (row / TOKENS_OUT) % HEAD_COUNT;
            uint b = row / (TOKENS_OUT * HEAD_COUNT);
            uint h = uint(HEAD_START) + local_h;
            uint q_token = uint(Q_TOKEN_START) + token_offset;
            uint qb = q_token / uint(BLOCK);
            uint q_base = (((b * H + h) * S + q_token) * D);
            uint q_mean_base = (((b * H + h) * NB + qb) * D);
            uint threshold_idx = ((b * H + h) * NB + qb);
            float threshold = thresholds[threshold_idx];
            float scale = static_cast<float>(SCALE_NUM) / static_cast<float>(SCALE_DEN);
            bool force_dense_query = q_token < uint(PREFIX_QUERY_TOKENS);

            float acc[D];
            for (uint d = 0; d < D; ++d) {
                acc[d] = 0.0f;
            }
            float row_max = -3.4028234663852886e38f;
            float row_sum = 0.0f;

            for (uint kb = 0; kb < NB; ++kb) {
                uint k_mean_base = (((b * H + h) * NB + kb) * D);
                bool sink_exact = (kb >= SINK_BLOCK_START) && (kb < SINK_BLOCK_STOP);
                uint distance = qb > kb ? (qb - kb) : (kb - qb);
                bool neighbor_exact = distance <= 1;
                bool exact_block = force_dense_query || sink_exact || neighbor_exact;

                if (!exact_block) {
                    float proxy_dot = 0.0f;
                    for (uint d = 0; d < D; ++d) {
                        proxy_dot += q_mean[q_mean_base + d] * k_mean[k_mean_base + d];
                    }
                    exact_block = (proxy_dot * scale) > threshold;
                }

                uint k_start = kb * uint(BLOCK);
                uint k_stop = metal::min(k_start + uint(BLOCK), uint(S));
                uint block_len = k_stop - k_start;

                if (exact_block) {
                    for (uint kt = k_start; kt < k_stop; ++kt) {
                        uint k_base = (((b * H + h) * S + kt) * D);
                        float dot = 0.0f;
                        for (uint d = 0; d < D; ++d) {
                            dot += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                        }
                        float score = dot * scale;
                        float new_max = metal::max(row_max, score);
                        float old_scale = metal::exp(row_max - new_max);
                        float item_scale = metal::exp(score - new_max);
                        uint v_base = (((b * H + h) * S + kt) * D);
                        for (uint d = 0; d < D; ++d) {
                            acc[d] = acc[d] * old_scale + item_scale * static_cast<float>(v[v_base + d]);
                        }
                        row_sum = row_sum * old_scale + item_scale;
                        row_max = new_max;
                    }
                } else {
                    float approx_dot = 0.0f;
                    for (uint d = 0; d < D; ++d) {
                        approx_dot += static_cast<float>(q[q_base + d]) * k_mean[k_mean_base + d];
                    }
                    float score = approx_dot * scale;
                    float new_max = metal::max(row_max, score);
                    float old_scale = metal::exp(row_max - new_max);
                    float item_scale = metal::exp(score - new_max);
                    uint vsum_base = (((b * H + h) * NB + kb) * D);
                    for (uint d = 0; d < D; ++d) {
                        acc[d] = acc[d] * old_scale + item_scale * v_sum[vsum_base + d];
                    }
                    row_sum = row_sum * old_scale + item_scale * static_cast<float>(block_len);
                    row_max = new_max;
                }
            }

            uint out_base = (((b * HEAD_COUNT + local_h) * TOKENS_OUT + token_offset) * D);
            float inv_sum = 1.0f / row_sum;
            for (uint d = 0; d < D; ++d) {
                out[out_base + d] = acc[d] * inv_sum;
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _forward_subset_kernel_v2():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_sparse_online_subset_v2",
        input_names=["q", "k", "v", "q_mean", "k_mean", "v_sum", "thresholds"],
        output_names=["out"],
        source=r"""
            uint row_group = threadgroup_position_in_grid.y;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            uint row = row_group * 4 + simd_group_id;
            if (row >= TOTAL_ROWS) {
                return;
            }

            uint token_offset = row % TOKENS_OUT;
            uint local_h = (row / TOKENS_OUT) % HEAD_COUNT;
            uint b = row / (TOKENS_OUT * HEAD_COUNT);
            uint h = uint(HEAD_START) + local_h;
            uint q_token = uint(Q_TOKEN_START) + token_offset;
            uint qb = q_token / uint(BLOCK);
            uint q_base = (((b * H + h) * S + q_token) * D);
            uint q_mean_base = (((b * H + h) * NB + qb) * D);
            uint threshold_idx = ((b * H + h) * NB + qb);
            float threshold = thresholds[threshold_idx];
            float scale = static_cast<float>(SCALE_NUM) / static_cast<float>(SCALE_DEN);
            bool force_dense_query = q_token < uint(PREFIX_QUERY_TOKENS);

            float row_max = -3.4028234663852886e38f;

            // Pass 1: each SIMD-group owns one query row.  Lanes cooperatively reduce Q·K over
            // D by striding lane, lane+32, ... so no dense scores or routing tensors are stored.
            for (uint kb = 0; kb < NB; ++kb) {
                uint k_mean_base = (((b * H + h) * NB + kb) * D);
                bool sink_exact = (kb >= SINK_BLOCK_START) && (kb < SINK_BLOCK_STOP);
                uint distance = qb > kb ? (qb - kb) : (kb - qb);
                bool neighbor_exact = distance <= 1;
                bool exact_block = force_dense_query || sink_exact || neighbor_exact;

                if (!exact_block) {
                    float proxy_part = 0.0f;
                    for (uint d = simd_lane_id; d < D; d += 32) {
                        proxy_part += q_mean[q_mean_base + d] * k_mean[k_mean_base + d];
                    }
                    float proxy_dot = simd_sum(proxy_part);
                    exact_block = (proxy_dot * scale) > threshold;
                }

                uint k_start = kb * uint(BLOCK);
                uint k_stop = metal::min(k_start + uint(BLOCK), uint(S));

                if (exact_block) {
                    for (uint kt = k_start; kt < k_stop; ++kt) {
                        uint k_base = (((b * H + h) * S + kt) * D);
                        float dot_part = 0.0f;
                        for (uint d = simd_lane_id; d < D; d += 32) {
                            dot_part += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                        }
                        float dot = simd_sum(dot_part);
                        row_max = metal::max(row_max, dot * scale);
                    }
                } else {
                    float dot_part = 0.0f;
                    for (uint d = simd_lane_id; d < D; d += 32) {
                        dot_part += static_cast<float>(q[q_base + d]) * k_mean[k_mean_base + d];
                    }
                    float dot = simd_sum(dot_part);
                    row_max = metal::max(row_max, dot * scale);
                }
            }

            float row_sum = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            uint d0 = simd_lane_id;
            uint d1 = simd_lane_id + 32;
            uint d2 = simd_lane_id + 64;
            uint d3 = simd_lane_id + 96;

            // Pass 2: accumulate the common denominator and four strided output coordinates per
            // lane.  Blocks not selected by threshold/sink/neighbor remain centroid/V-sum approx.
            for (uint kb = 0; kb < NB; ++kb) {
                uint k_mean_base = (((b * H + h) * NB + kb) * D);
                bool sink_exact = (kb >= SINK_BLOCK_START) && (kb < SINK_BLOCK_STOP);
                uint distance = qb > kb ? (qb - kb) : (kb - qb);
                bool neighbor_exact = distance <= 1;
                bool exact_block = force_dense_query || sink_exact || neighbor_exact;

                if (!exact_block) {
                    float proxy_part = 0.0f;
                    for (uint d = simd_lane_id; d < D; d += 32) {
                        proxy_part += q_mean[q_mean_base + d] * k_mean[k_mean_base + d];
                    }
                    float proxy_dot = simd_sum(proxy_part);
                    exact_block = (proxy_dot * scale) > threshold;
                }

                uint k_start = kb * uint(BLOCK);
                uint k_stop = metal::min(k_start + uint(BLOCK), uint(S));
                uint block_len = k_stop - k_start;

                if (exact_block) {
                    for (uint kt = k_start; kt < k_stop; ++kt) {
                        uint k_base = (((b * H + h) * S + kt) * D);
                        float dot_part = 0.0f;
                        for (uint d = simd_lane_id; d < D; d += 32) {
                            dot_part += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                        }
                        float dot = simd_sum(dot_part);
                        float weight = metal::exp(dot * scale - row_max);
                        row_sum += weight;
                        uint v_base = (((b * H + h) * S + kt) * D);
                        if (d0 < D) { acc0 += weight * static_cast<float>(v[v_base + d0]); }
                        if (d1 < D) { acc1 += weight * static_cast<float>(v[v_base + d1]); }
                        if (d2 < D) { acc2 += weight * static_cast<float>(v[v_base + d2]); }
                        if (d3 < D) { acc3 += weight * static_cast<float>(v[v_base + d3]); }
                    }
                } else {
                    float dot_part = 0.0f;
                    for (uint d = simd_lane_id; d < D; d += 32) {
                        dot_part += static_cast<float>(q[q_base + d]) * k_mean[k_mean_base + d];
                    }
                    float dot = simd_sum(dot_part);
                    float weight = metal::exp(dot * scale - row_max);
                    row_sum += weight * static_cast<float>(block_len);
                    uint vsum_base = (((b * H + h) * NB + kb) * D);
                    if (d0 < D) { acc0 += weight * v_sum[vsum_base + d0]; }
                    if (d1 < D) { acc1 += weight * v_sum[vsum_base + d1]; }
                    if (d2 < D) { acc2 += weight * v_sum[vsum_base + d2]; }
                    if (d3 < D) { acc3 += weight * v_sum[vsum_base + d3]; }
                }
            }

            uint out_base = (((b * HEAD_COUNT + local_h) * TOKENS_OUT + token_offset) * D);
            float inv_sum = 1.0f / row_sum;
            if (d0 < D) { out[out_base + d0] = acc0 * inv_sum; }
            if (d1 < D) { out[out_base + d1] = acc1 * inv_sum; }
            if (d2 < D) { out[out_base + d2] = acc2 * inv_sum; }
            if (d3 < D) { out[out_base + d3] = acc3 * inv_sum; }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _forward_subset_kernel_v3():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_sparse_online_subset_v3_qblock_tiled",
        input_names=["q", "k", "v", "q_mean", "k_mean", "v_sum", "thresholds"],
        output_names=["out"],
        source=r"""
            uint group = threadgroup_position_in_grid.y;
            uint row_in_block = thread_index_in_threadgroup;
            uint local_qb = group % Q_BLOCK_COUNT;
            uint local_h = (group / Q_BLOCK_COUNT) % HEAD_COUNT;
            uint b = group / (Q_BLOCK_COUNT * HEAD_COUNT);
            uint qb = uint(Q_BLOCK_START) + local_qb;
            uint h = uint(HEAD_START) + local_h;
            uint q_token = qb * uint(BLOCK) + row_in_block;
            uint q_mean_base = (((b * H + h) * NB + qb) * D);
            uint threshold_idx = ((b * H + h) * NB + qb);
            float threshold = thresholds[threshold_idx];
            float scale = static_cast<float>(SCALE_NUM) / static_cast<float>(SCALE_DEN);

            // The route is per query block, not per query row.  Compute it once for the whole
            // 64-row tile, then reuse it from all row threads.  This avoids the v1/v2 repeated
            // centroid routing work while preserving strict threshold + sink + neighbor semantics.
            threadgroup ushort exact_flags[NB];
            if (row_in_block == 0) {
                for (uint kb = 0; kb < NB; ++kb) {
                    uint k_mean_base = (((b * H + h) * NB + kb) * D);
                    bool sink_exact = (kb >= SINK_BLOCK_START) && (kb < SINK_BLOCK_STOP);
                    uint distance = qb > kb ? (qb - kb) : (kb - qb);
                    bool neighbor_exact = distance <= 1;
                    bool exact_block = sink_exact || neighbor_exact;
                    if (!exact_block) {
                        float proxy_dot = 0.0f;
                        for (uint d = 0; d < D; ++d) {
                            proxy_dot += q_mean[q_mean_base + d] * k_mean[k_mean_base + d];
                        }
                        exact_block = (proxy_dot * scale) > threshold;
                    }
                    exact_flags[kb] = exact_block ? ushort(1) : ushort(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (q_token >= uint(S)) {
                return;
            }
            uint token_offset = q_token - uint(Q_TOKEN_START);
            if (token_offset >= uint(TOKENS_OUT)) {
                return;
            }
            uint q_base = (((b * H + h) * S + q_token) * D);
            bool force_dense_query = q_token < uint(PREFIX_QUERY_TOKENS);

            float acc[D];
            for (uint d = 0; d < D; ++d) {
                acc[d] = 0.0f;
            }
            float row_max = -3.4028234663852886e38f;
            float row_sum = 0.0f;

            for (uint kb = 0; kb < NB; ++kb) {
                uint k_start = kb * uint(BLOCK);
                uint k_stop = metal::min(k_start + uint(BLOCK), uint(S));
                uint block_len = k_stop - k_start;
                bool exact_block = force_dense_query || (exact_flags[kb] != ushort(0));

                if (exact_block) {
                    for (uint kt = k_start; kt < k_stop; ++kt) {
                        uint k_base = (((b * H + h) * S + kt) * D);
                        float dot = 0.0f;
                        for (uint d = 0; d < D; ++d) {
                            dot += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                        }
                        float score = dot * scale;
                        float new_max = metal::max(row_max, score);
                        float old_scale = metal::exp(row_max - new_max);
                        float item_scale = metal::exp(score - new_max);
                        uint v_base = (((b * H + h) * S + kt) * D);
                        for (uint d = 0; d < D; ++d) {
                            acc[d] = acc[d] * old_scale + item_scale * static_cast<float>(v[v_base + d]);
                        }
                        row_sum = row_sum * old_scale + item_scale;
                        row_max = new_max;
                    }
                } else {
                    uint k_mean_base = (((b * H + h) * NB + kb) * D);
                    float approx_dot = 0.0f;
                    for (uint d = 0; d < D; ++d) {
                        approx_dot += static_cast<float>(q[q_base + d]) * k_mean[k_mean_base + d];
                    }
                    float score = approx_dot * scale;
                    float new_max = metal::max(row_max, score);
                    float old_scale = metal::exp(row_max - new_max);
                    float item_scale = metal::exp(score - new_max);
                    uint vsum_base = (((b * H + h) * NB + kb) * D);
                    for (uint d = 0; d < D; ++d) {
                        acc[d] = acc[d] * old_scale + item_scale * v_sum[vsum_base + d];
                    }
                    row_sum = row_sum * old_scale + item_scale * static_cast<float>(block_len);
                    row_max = new_max;
                }
            }

            uint out_base = (((b * HEAD_COUNT + local_h) * TOKENS_OUT + token_offset) * D);
            float inv_sum = 1.0f / row_sum;
            for (uint d = 0; d < D; ++d) {
                out[out_base + d] = acc[d] * inv_sum;
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _forward_subset_kernel_v4():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_sparse_online_subset_v4_qblock_simd",
        input_names=["q", "k", "v", "q_mean", "k_mean", "v_sum", "thresholds"],
        output_names=["out"],
        source=r"""
            uint group = threadgroup_position_in_grid.y;
            uint tid = thread_index_in_threadgroup;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            uint local_qb = group % Q_BLOCK_COUNT;
            uint local_h = (group / Q_BLOCK_COUNT) % HEAD_COUNT;
            uint b = group / (Q_BLOCK_COUNT * HEAD_COUNT);
            uint qb = uint(Q_BLOCK_START) + local_qb;
            uint h = uint(HEAD_START) + local_h;
            uint q_mean_base = (((b * H + h) * NB + qb) * D);
            uint threshold_idx = ((b * H + h) * NB + qb);
            float threshold = thresholds[threshold_idx];
            float scale = static_cast<float>(SCALE_NUM) / static_cast<float>(SCALE_DEN);

            // Route is per query block/head tile.  Unlike v2, do not recompute centroid routing
            // per query row; unlike v3, exact-token QK and output coordinates are reduced by
            // SIMD-groups across D.  The route is transient threadgroup state, not a persistent
            // routing tensor or dense mask.
            threadgroup ushort exact_flags[NB];
            if (tid == 0) {
                for (uint kb = 0; kb < NB; ++kb) {
                    uint k_mean_base = (((b * H + h) * NB + kb) * D);
                    bool sink_exact = (kb >= SINK_BLOCK_START) && (kb < SINK_BLOCK_STOP);
                    uint distance = qb > kb ? (qb - kb) : (kb - qb);
                    bool neighbor_exact = distance <= 1;
                    bool exact_block = sink_exact || neighbor_exact;
                    if (!exact_block) {
                        float proxy_dot = 0.0f;
                        for (uint d = 0; d < D; ++d) {
                            proxy_dot += q_mean[q_mean_base + d] * k_mean[k_mean_base + d];
                        }
                        exact_block = (proxy_dot * scale) > threshold;
                    }
                    exact_flags[kb] = exact_block ? ushort(1) : ushort(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Eight SIMD-groups in the threadgroup process the 64 query rows in waves.  Each
            // SIMD-group owns one row at a time; lanes reduce QK over D and write four strided
            // output coordinates (lane, lane+32, lane+64, lane+96) for D<=128.
            for (uint row_base = 0; row_base < uint(BLOCK); row_base += 8) {
                uint row_in_block = row_base + simd_group_id;
                if (row_in_block >= uint(BLOCK)) {
                    continue;
                }
                uint q_token = qb * uint(BLOCK) + row_in_block;
                if (q_token >= uint(S)) {
                    continue;
                }
                uint token_offset = q_token - uint(Q_TOKEN_START);
                if (token_offset >= uint(TOKENS_OUT)) {
                    continue;
                }

                uint q_base = (((b * H + h) * S + q_token) * D);
                bool force_dense_query = q_token < uint(PREFIX_QUERY_TOKENS);
                float row_max = -3.4028234663852886e38f;

                for (uint kb = 0; kb < NB; ++kb) {
                    uint k_start = kb * uint(BLOCK);
                    uint k_stop = metal::min(k_start + uint(BLOCK), uint(S));
                    bool exact_block = force_dense_query || (exact_flags[kb] != ushort(0));

                    if (exact_block) {
                        for (uint kt = k_start; kt < k_stop; ++kt) {
                            uint k_base = (((b * H + h) * S + kt) * D);
                            float dot_part = 0.0f;
                            for (uint d = simd_lane_id; d < D; d += 32) {
                                dot_part += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                            }
                            float dot = simd_sum(dot_part);
                            row_max = metal::max(row_max, dot * scale);
                        }
                    } else {
                        uint k_mean_base = (((b * H + h) * NB + kb) * D);
                        float dot_part = 0.0f;
                        for (uint d = simd_lane_id; d < D; d += 32) {
                            dot_part += static_cast<float>(q[q_base + d]) * k_mean[k_mean_base + d];
                        }
                        float dot = simd_sum(dot_part);
                        row_max = metal::max(row_max, dot * scale);
                    }
                }

                float row_sum = 0.0f;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                uint d0 = simd_lane_id;
                uint d1 = simd_lane_id + 32;
                uint d2 = simd_lane_id + 64;
                uint d3 = simd_lane_id + 96;

                for (uint kb = 0; kb < NB; ++kb) {
                    uint k_start = kb * uint(BLOCK);
                    uint k_stop = metal::min(k_start + uint(BLOCK), uint(S));
                    uint block_len = k_stop - k_start;
                    bool exact_block = force_dense_query || (exact_flags[kb] != ushort(0));

                    if (exact_block) {
                        for (uint kt = k_start; kt < k_stop; ++kt) {
                            uint k_base = (((b * H + h) * S + kt) * D);
                            float dot_part = 0.0f;
                            for (uint d = simd_lane_id; d < D; d += 32) {
                                dot_part += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                            }
                            float dot = simd_sum(dot_part);
                            float weight = metal::exp(dot * scale - row_max);
                            row_sum += weight;
                            uint v_base = (((b * H + h) * S + kt) * D);
                            if (d0 < D) { acc0 += weight * static_cast<float>(v[v_base + d0]); }
                            if (d1 < D) { acc1 += weight * static_cast<float>(v[v_base + d1]); }
                            if (d2 < D) { acc2 += weight * static_cast<float>(v[v_base + d2]); }
                            if (d3 < D) { acc3 += weight * static_cast<float>(v[v_base + d3]); }
                        }
                    } else {
                        uint k_mean_base = (((b * H + h) * NB + kb) * D);
                        float dot_part = 0.0f;
                        for (uint d = simd_lane_id; d < D; d += 32) {
                            dot_part += static_cast<float>(q[q_base + d]) * k_mean[k_mean_base + d];
                        }
                        float dot = simd_sum(dot_part);
                        float weight = metal::exp(dot * scale - row_max);
                        row_sum += weight * static_cast<float>(block_len);
                        uint vsum_base = (((b * H + h) * NB + kb) * D);
                        if (d0 < D) { acc0 += weight * v_sum[vsum_base + d0]; }
                        if (d1 < D) { acc1 += weight * v_sum[vsum_base + d1]; }
                        if (d2 < D) { acc2 += weight * v_sum[vsum_base + d2]; }
                        if (d3 < D) { acc3 += weight * v_sum[vsum_base + d3]; }
                    }
                }

                uint out_base = (((b * HEAD_COUNT + local_h) * TOKENS_OUT + token_offset) * D);
                float inv_sum = 1.0f / row_sum;
                if (d0 < D) { out[out_base + d0] = acc0 * inv_sum; }
                if (d1 < D) { out[out_base + d1] = acc1 * inv_sum; }
                if (d2 < D) { out[out_base + d2] = acc2 * inv_sum; }
                if (d3 < D) { out[out_base + d3] = acc3 * inv_sum; }
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def _exact_tile_kernel_v5():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_macsol_exact_tile_v5",
        input_names=["q", "k", "v"],
        output_names=["out"],
        source=r"""
            uint group = threadgroup_position_in_grid.y;
            uint tid = thread_index_in_threadgroup;
            uint local_kv = group % KV_BLOCK_COUNT;
            uint local_h = (group / KV_BLOCK_COUNT) % HEAD_COUNT;
            uint b = group / (KV_BLOCK_COUNT * HEAD_COUNT);
            uint h = uint(HEAD_START) + local_h;
            uint q_start = uint(Q_BLOCK_START) * uint(BLOCK);
            uint k_start = (uint(KV_BLOCK_START) + local_kv) * uint(BLOCK);
            float scale = static_cast<float>(SCALE_NUM) / static_cast<float>(SCALE_DEN);

            // One full 64x64 exact-attention tile.  The score tile and row softmax state are
            // transient threadgroup storage only; no dense SxS matrix, mask, or route tensor is
            // materialized outside this threadgroup.
            threadgroup float scores[BLOCK * BLOCK];
            threadgroup float row_max[BLOCK];
            threadgroup float row_sum[BLOCK];

            for (uint idx = tid; idx < uint(BLOCK) * uint(BLOCK); idx += uint(THREADS)) {
                uint qi = idx / uint(BLOCK);
                uint kj = idx - qi * uint(BLOCK);
                uint q_base = (((b * H + h) * S + (q_start + qi)) * D);
                uint k_base = (((b * H + h) * S + (k_start + kj)) * D);
                float dot = 0.0f;
                for (uint d = 0; d < D; ++d) {
                    dot += static_cast<float>(q[q_base + d]) * static_cast<float>(k[k_base + d]);
                }
                scores[idx] = dot * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint qi = tid; qi < uint(BLOCK); qi += uint(THREADS)) {
                float m = -3.4028234663852886e38f;
                uint row_base = qi * uint(BLOCK);
                for (uint kj = 0; kj < uint(BLOCK); ++kj) {
                    m = metal::max(m, scores[row_base + kj]);
                }
                float s = 0.0f;
                for (uint kj = 0; kj < uint(BLOCK); ++kj) {
                    s += metal::exp(scores[row_base + kj] - m);
                }
                row_max[qi] = m;
                row_sum[qi] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint idx = tid; idx < uint(BLOCK) * uint(D); idx += uint(THREADS)) {
                uint qi = idx / uint(D);
                uint d = idx - qi * uint(D);
                uint row_base = qi * uint(BLOCK);
                float acc = 0.0f;
                float m = row_max[qi];
                for (uint kj = 0; kj < uint(BLOCK); ++kj) {
                    float weight = metal::exp(scores[row_base + kj] - m);
                    uint v_base = (((b * H + h) * S + (k_start + kj)) * D);
                    acc += weight * static_cast<float>(v[v_base + d]);
                }
                uint out_base = ((((b * HEAD_COUNT + local_h) * KV_BLOCK_COUNT + local_kv) * BLOCK + qi) * D);
                out[out_base + d] = acc / row_sum[qi];
            }
        """,
        compile_options={"math_mode": "safe"},
    )


def has_macsol_metal() -> bool:
    """Return whether this MLX build exposes the scalar provenance custom Metal path."""

    try:
        return _summary_kernel() is not None and _threshold_kernel() is not None and _forward_subset_kernel() is not None
    except Exception:
        return False


def has_macsol_metal_v2() -> bool:
    """Return whether this MLX build exposes the rejected SIMD-group cooperative v2 Metal path."""

    try:
        return _summary_kernel() is not None and _threshold_kernel() is not None and _forward_subset_kernel_v2() is not None
    except Exception:
        return False


def has_macsol_metal_v3() -> bool:
    """Return whether this MLX build exposes the rejected q-block tiled v3 Metal path."""

    try:
        return _summary_kernel() is not None and _threshold_kernel() is not None and _forward_subset_kernel_v3() is not None
    except Exception:
        return False


def has_macsol_metal_v4() -> bool:
    """Return whether this MLX build exposes the rejected q-block/SIMD-group v4 Metal path."""

    try:
        return _summary_kernel() is not None and _threshold_kernel() is not None and _forward_subset_kernel_v4() is not None
    except Exception:
        return False


def has_macsol_metal_v5() -> bool:
    """Return whether this MLX build exposes the default-off v5 exact-tile microkernel."""

    try:
        return _exact_tile_kernel_v5() is not None
    except Exception:
        return False


def _validate_qkv(q: mx.array, k: mx.array, v: mx.array) -> tuple[int, int, int, int]:
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise ValueError(f"q/k/v must all be rank-4 [B,H,S,D], got {q.shape}, {k.shape}, {v.shape}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"self-attention requires identical q/k/v shapes, got {q.shape}, {k.shape}, {v.shape}")
    batch, heads, sequence, head_dim = (int(dim) for dim in q.shape)
    if batch <= 0 or heads <= 0 or sequence <= 0 or head_dim <= 0:
        raise ValueError(f"q/k/v dimensions must be positive, got {q.shape}")
    return batch, heads, sequence, head_dim


def macsol_prepare_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    block_size: int = 64,
    tau: float = 1.0,
    scale: float | None = None,
) -> MacSolMetalPreparation:
    """Prepare Q/K means, V sums, and per-query-block thresholds with Metal kernels."""

    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    scale = float(head_dim ** -0.5 if scale is None else scale)
    block_count = int(math.ceil(sequence / block_size))
    summary_kernel = _summary_kernel()
    threshold_kernel = _threshold_kernel()
    if summary_kernel is None or threshold_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")

    summary_shape = (batch, heads, block_count, head_dim)
    q_mean, k_mean, v_sum = summary_kernel(
        inputs=[q, k, v],
        template=[
            ("T", q.dtype),
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("NB", block_count),
            ("BLOCK", block_size),
        ],
        grid=(batch * heads * block_count * head_dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[summary_shape, summary_shape, summary_shape],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    scale_num, scale_den = _ratio_template(scale)
    tau_num, tau_den = _ratio_template(float(tau))
    threshold_shape = (batch, heads, block_count)
    thresholds, proxy_mean, proxy_std = threshold_kernel(
        inputs=[q_mean, k_mean],
        template=[
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("NB", block_count),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
            ("TAU_NUM", tau_num),
            ("TAU_DEN", tau_den),
        ],
        grid=(batch * heads * block_count, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[threshold_shape, threshold_shape, threshold_shape],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return MacSolMetalPreparation(
        q_mean=q_mean,
        k_mean=k_mean,
        v_sum=v_sum,
        thresholds=thresholds,
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        block_count=block_count,
        block_size=block_size,
        scale=scale,
        tau=float(tau),
    )


def macsol_attention_metal_subset(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: MacSolReferenceConfig | None = None,
    *,
    q_block_start: int = 0,
    q_block_count: int | None = None,
    head_start: int = 0,
    head_count: int | None = None,
    scale: float | None = None,
    preparation: MacSolMetalPreparation | None = None,
) -> MacSolMetalAttentionResult:
    """Run the rejected scalar row-thread sparse online-softmax Metal probe on a subset.

    The returned ``output`` has shape ``[B, head_count, tokens_out, D]`` where ``tokens_out`` is
    the exact number of query tokens covered by the requested block range.  Only threshold routing
    is implemented here; rejected ``budget_topk``, ``threshold_h3_structure``, and spatial-tube
    variants are deliberately rejected so this provenance probe cannot silently become a new
    heuristic or speed path.
    """

    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    config = config or MacSolReferenceConfig()
    config.validate_for_sequence(sequence)
    if str(config.routing_mode) != "threshold":
        raise ValueError("MacSol Metal v1 archive probe implements source-faithful threshold routing only")
    block_size = int(config.block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    scale = float(head_dim ** -0.5 if scale is None else scale)
    block_count = int(math.ceil(sequence / block_size))

    q_block_start = int(q_block_start)
    if q_block_start < 0 or q_block_start >= block_count:
        raise ValueError(f"q_block_start must be in [0,{block_count}), got {q_block_start}")
    if q_block_count is None:
        q_block_count = block_count - q_block_start
    q_block_count = int(q_block_count)
    if q_block_count <= 0 or q_block_start + q_block_count > block_count:
        raise ValueError(
            f"q_block_count must keep range inside [0,{block_count}], got start={q_block_start} count={q_block_count}"
        )
    head_start = int(head_start)
    if head_start < 0 or head_start >= heads:
        raise ValueError(f"head_start must be in [0,{heads}), got {head_start}")
    if head_count is None:
        head_count = heads - head_start
    head_count = int(head_count)
    if head_count <= 0 or head_start + head_count > heads:
        raise ValueError(f"head_count must keep range inside [0,{heads}], got start={head_start} count={head_count}")

    if preparation is None:
        preparation = macsol_prepare_metal(q, k, v, block_size=block_size, tau=float(config.tau), scale=scale)
    elif (
        preparation.block_count != block_count
        or preparation.block_size != block_size
        or abs(float(preparation.scale) - scale) > 1e-12
        or abs(float(preparation.tau) - float(config.tau)) > 1e-12
    ):
        raise ValueError("provided MacSol Metal preparation does not match q/k/v shape, block size, scale, or tau")

    forward_kernel = _forward_subset_kernel()
    if forward_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")

    rounded_sink_tokens, rounded_sink_block_range = rounded_sink_blocks(
        sequence,
        sink_start=int(config.sink_start),
        sink_tokens=int(config.sink_tokens),
        block_size=block_size,
    )
    del rounded_sink_tokens
    q_token_start = q_block_start * block_size
    q_token_stop = min(sequence, (q_block_start + q_block_count) * block_size)
    tokens_out = q_token_stop - q_token_start
    if tokens_out <= 0:
        raise ValueError("requested query block subset is empty")

    scale_num, scale_den = _ratio_template(scale)
    output_shape = (batch, head_count, tokens_out, head_dim)
    output = forward_kernel(
        inputs=[q, k, v, preparation.q_mean, preparation.k_mean, preparation.v_sum, preparation.thresholds],
        template=[
            ("T", q.dtype),
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("NB", block_count),
            ("BLOCK", block_size),
            ("HEAD_START", head_start),
            ("HEAD_COUNT", head_count),
            ("Q_TOKEN_START", q_token_start),
            ("TOKENS_OUT", tokens_out),
            ("PREFIX_QUERY_TOKENS", int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0)),
            ("SINK_BLOCK_START", int(rounded_sink_block_range[0])),
            ("SINK_BLOCK_STOP", int(rounded_sink_block_range[1])),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
        ],
        grid=(batch * head_count * tokens_out, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    return MacSolMetalAttentionResult(
        output=output,
        preparation=preparation,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        q_token_range=(q_token_start, q_token_stop),
        rounded_sink_block_range=(int(rounded_sink_block_range[0]), int(rounded_sink_block_range[1])),
        prefix_query_tokens=int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0),
    )


def macsol_attention_metal_subset_v2(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: MacSolReferenceConfig | None = None,
    *,
    q_block_start: int = 0,
    q_block_count: int | None = None,
    head_start: int = 0,
    head_count: int | None = None,
    scale: float | None = None,
    preparation: MacSolMetalPreparation | None = None,
) -> MacSolMetalAttentionResult:
    """Run the rejected/provenance-only SIMD-group cooperative v2 sparse online-softmax Metal probe.

    The kernel maps four query rows to one 128-thread Metal threadgroup, with one SIMD-group per
    row.  Lanes cooperatively reduce each Q·K score over the head dimension, then accumulate
    strided output coordinates.  The path preserves the source-faithful threshold, neighbor,
    rounded sink, prefix-query dense/exact, and centroid/V-sum approximation semantics, and it still
    avoids dense SxS scores, dense masks, and persistent routing tensors.
    """

    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    config = config or MacSolReferenceConfig()
    config.validate_for_sequence(sequence)
    if str(config.routing_mode) != "threshold":
        raise ValueError("MacSol Metal v2 archive probe implements source-faithful threshold routing only")
    if int(head_dim) > 128:
        raise ValueError(f"MacSol Metal v2 supports head_dim <= 128 for one-dimension-per-thread output, got {head_dim}")
    block_size = int(config.block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    scale = float(head_dim ** -0.5 if scale is None else scale)
    block_count = int(math.ceil(sequence / block_size))

    q_block_start = int(q_block_start)
    if q_block_start < 0 or q_block_start >= block_count:
        raise ValueError(f"q_block_start must be in [0,{block_count}), got {q_block_start}")
    if q_block_count is None:
        q_block_count = block_count - q_block_start
    q_block_count = int(q_block_count)
    if q_block_count <= 0 or q_block_start + q_block_count > block_count:
        raise ValueError(
            f"q_block_count must keep range inside [0,{block_count}], got start={q_block_start} count={q_block_count}"
        )
    head_start = int(head_start)
    if head_start < 0 or head_start >= heads:
        raise ValueError(f"head_start must be in [0,{heads}), got {head_start}")
    if head_count is None:
        head_count = heads - head_start
    head_count = int(head_count)
    if head_count <= 0 or head_start + head_count > heads:
        raise ValueError(f"head_count must keep range inside [0,{heads}], got start={head_start} count={head_count}")

    if preparation is None:
        preparation = macsol_prepare_metal(q, k, v, block_size=block_size, tau=float(config.tau), scale=scale)
    elif (
        preparation.block_count != block_count
        or preparation.block_size != block_size
        or abs(float(preparation.scale) - scale) > 1e-12
        or abs(float(preparation.tau) - float(config.tau)) > 1e-12
    ):
        raise ValueError("provided MacSol Metal preparation does not match q/k/v shape, block size, scale, or tau")

    forward_kernel = _forward_subset_kernel_v2()
    if forward_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")

    rounded_sink_tokens, rounded_sink_block_range = rounded_sink_blocks(
        sequence,
        sink_start=int(config.sink_start),
        sink_tokens=int(config.sink_tokens),
        block_size=block_size,
    )
    del rounded_sink_tokens
    q_token_start = q_block_start * block_size
    q_token_stop = min(sequence, (q_block_start + q_block_count) * block_size)
    tokens_out = q_token_stop - q_token_start
    if tokens_out <= 0:
        raise ValueError("requested query block subset is empty")

    scale_num, scale_den = _ratio_template(scale)
    output_shape = (batch, head_count, tokens_out, head_dim)
    total_rows = int(batch * head_count * tokens_out)
    output = forward_kernel(
        inputs=[q, k, v, preparation.q_mean, preparation.k_mean, preparation.v_sum, preparation.thresholds],
        template=[
            ("T", q.dtype),
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("NB", block_count),
            ("BLOCK", block_size),
            ("HEAD_START", head_start),
            ("HEAD_COUNT", head_count),
            ("Q_TOKEN_START", q_token_start),
            ("TOKENS_OUT", tokens_out),
            ("PREFIX_QUERY_TOKENS", int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0)),
            ("SINK_BLOCK_START", int(rounded_sink_block_range[0])),
            ("SINK_BLOCK_STOP", int(rounded_sink_block_range[1])),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
            ("TOTAL_ROWS", total_rows),
        ],
        grid=(128, int(math.ceil(total_rows / 4)), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    return MacSolMetalAttentionResult(
        output=output,
        preparation=preparation,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        q_token_range=(q_token_start, q_token_stop),
        rounded_sink_block_range=(int(rounded_sink_block_range[0]), int(rounded_sink_block_range[1])),
        prefix_query_tokens=int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0),
    )


def macsol_attention_metal_subset_v3(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: MacSolReferenceConfig | None = None,
    *,
    q_block_start: int = 0,
    q_block_count: int | None = None,
    head_start: int = 0,
    head_count: int | None = None,
    scale: float | None = None,
    preparation: MacSolMetalPreparation | None = None,
) -> MacSolMetalAttentionResult:
    """Run the rejected/default-off q-block tiled v3 sparse online-softmax Metal probe.

    One Metal threadgroup owns one ``[batch, head, query-block]`` tile.  The group computes the
    source-faithful exact/approximate KV-block route once into threadgroup memory, then up to
    ``block_size`` row threads reuse that route while accumulating the same online-softmax mixture
    of exact token blocks and centroid/V-sum approximate blocks.  This path remains an internal
    subset gate only and is not wired into generation.
    """

    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    config = config or MacSolReferenceConfig()
    config.validate_for_sequence(sequence)
    if str(config.routing_mode) != "threshold":
        raise ValueError("MacSol Metal v3 archive probe implements source-faithful threshold routing only")
    block_size = int(config.block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size > 1024:
        raise ValueError(f"MacSol Metal v3 requires block_size <= 1024 for one thread per query row, got {block_size}")
    scale = float(head_dim ** -0.5 if scale is None else scale)
    block_count = int(math.ceil(sequence / block_size))

    q_block_start = int(q_block_start)
    if q_block_start < 0 or q_block_start >= block_count:
        raise ValueError(f"q_block_start must be in [0,{block_count}), got {q_block_start}")
    if q_block_count is None:
        q_block_count = block_count - q_block_start
    q_block_count = int(q_block_count)
    if q_block_count <= 0 or q_block_start + q_block_count > block_count:
        raise ValueError(
            f"q_block_count must keep range inside [0,{block_count}], got start={q_block_start} count={q_block_count}"
        )
    head_start = int(head_start)
    if head_start < 0 or head_start >= heads:
        raise ValueError(f"head_start must be in [0,{heads}), got {head_start}")
    if head_count is None:
        head_count = heads - head_start
    head_count = int(head_count)
    if head_count <= 0 or head_start + head_count > heads:
        raise ValueError(f"head_count must keep range inside [0,{heads}], got start={head_start} count={head_count}")

    if preparation is None:
        preparation = macsol_prepare_metal(q, k, v, block_size=block_size, tau=float(config.tau), scale=scale)
    elif (
        preparation.block_count != block_count
        or preparation.block_size != block_size
        or abs(float(preparation.scale) - scale) > 1e-12
        or abs(float(preparation.tau) - float(config.tau)) > 1e-12
    ):
        raise ValueError("provided MacSol Metal preparation does not match q/k/v shape, block size, scale, or tau")

    forward_kernel = _forward_subset_kernel_v3()
    if forward_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")

    rounded_sink_tokens, rounded_sink_block_range = rounded_sink_blocks(
        sequence,
        sink_start=int(config.sink_start),
        sink_tokens=int(config.sink_tokens),
        block_size=block_size,
    )
    del rounded_sink_tokens
    q_token_start = q_block_start * block_size
    q_token_stop = min(sequence, (q_block_start + q_block_count) * block_size)
    tokens_out = q_token_stop - q_token_start
    if tokens_out <= 0:
        raise ValueError("requested query block subset is empty")

    scale_num, scale_den = _ratio_template(scale)
    output_shape = (batch, head_count, tokens_out, head_dim)
    output = forward_kernel(
        inputs=[q, k, v, preparation.q_mean, preparation.k_mean, preparation.v_sum, preparation.thresholds],
        template=[
            ("T", q.dtype),
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("NB", block_count),
            ("BLOCK", block_size),
            ("HEAD_START", head_start),
            ("HEAD_COUNT", head_count),
            ("Q_BLOCK_START", q_block_start),
            ("Q_BLOCK_COUNT", q_block_count),
            ("Q_TOKEN_START", q_token_start),
            ("TOKENS_OUT", tokens_out),
            ("PREFIX_QUERY_TOKENS", int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0)),
            ("SINK_BLOCK_START", int(rounded_sink_block_range[0])),
            ("SINK_BLOCK_STOP", int(rounded_sink_block_range[1])),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
        ],
        grid=(block_size, batch * head_count * q_block_count, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    return MacSolMetalAttentionResult(
        output=output,
        preparation=preparation,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        q_token_range=(q_token_start, q_token_stop),
        rounded_sink_block_range=(int(rounded_sink_block_range[0]), int(rounded_sink_block_range[1])),
        prefix_query_tokens=int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0),
    )


def macsol_attention_metal_subset_v4(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: MacSolReferenceConfig | None = None,
    *,
    q_block_start: int = 0,
    q_block_count: int | None = None,
    head_start: int = 0,
    head_count: int | None = None,
    scale: float | None = None,
    preparation: MacSolMetalPreparation | None = None,
) -> MacSolMetalAttentionResult:
    """Run the rejected/provenance-only q-block/SIMD-group cooperative v4 sparse Metal probe.

    One Metal threadgroup owns one ``[batch, head, query-block]`` tile, computes the threshold /
    sink / neighbor route once in threadgroup memory, then eight SIMD-groups sweep the 64 query rows
    in waves.  Within a row wave, lanes cooperatively reduce exact-token QK scores over the head
    dimension and accumulate four strided output coordinates per lane.  The path preserves the
    same source-faithful threshold routing and centroid/V-sum approximation as the reference, while
    avoiding dense SxS scores, dense masks, and persistent routing tensors.
    """

    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    config = config or MacSolReferenceConfig()
    config.validate_for_sequence(sequence)
    if str(config.routing_mode) != "threshold":
        raise ValueError("MacSol Metal v4 archive probe implements source-faithful threshold routing only")
    if int(head_dim) > 128:
        raise ValueError(f"MacSol Metal v4 supports head_dim <= 128 for four strided dims per SIMD lane, got {head_dim}")
    block_size = int(config.block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size > 1024:
        raise ValueError(f"MacSol Metal v4 requires block_size <= 1024 for one query-block threadgroup, got {block_size}")
    scale = float(head_dim ** -0.5 if scale is None else scale)
    block_count = int(math.ceil(sequence / block_size))

    q_block_start = int(q_block_start)
    if q_block_start < 0 or q_block_start >= block_count:
        raise ValueError(f"q_block_start must be in [0,{block_count}), got {q_block_start}")
    if q_block_count is None:
        q_block_count = block_count - q_block_start
    q_block_count = int(q_block_count)
    if q_block_count <= 0 or q_block_start + q_block_count > block_count:
        raise ValueError(
            f"q_block_count must keep range inside [0,{block_count}], got start={q_block_start} count={q_block_count}"
        )
    head_start = int(head_start)
    if head_start < 0 or head_start >= heads:
        raise ValueError(f"head_start must be in [0,{heads}), got {head_start}")
    if head_count is None:
        head_count = heads - head_start
    head_count = int(head_count)
    if head_count <= 0 or head_start + head_count > heads:
        raise ValueError(f"head_count must keep range inside [0,{heads}], got start={head_start} count={head_count}")

    if preparation is None:
        preparation = macsol_prepare_metal(q, k, v, block_size=block_size, tau=float(config.tau), scale=scale)
    elif (
        preparation.block_count != block_count
        or preparation.block_size != block_size
        or abs(float(preparation.scale) - scale) > 1e-12
        or abs(float(preparation.tau) - float(config.tau)) > 1e-12
    ):
        raise ValueError("provided MacSol Metal preparation does not match q/k/v shape, block size, scale, or tau")

    forward_kernel = _forward_subset_kernel_v4()
    if forward_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")

    rounded_sink_tokens, rounded_sink_block_range = rounded_sink_blocks(
        sequence,
        sink_start=int(config.sink_start),
        sink_tokens=int(config.sink_tokens),
        block_size=block_size,
    )
    del rounded_sink_tokens
    q_token_start = q_block_start * block_size
    q_token_stop = min(sequence, (q_block_start + q_block_count) * block_size)
    tokens_out = q_token_stop - q_token_start
    if tokens_out <= 0:
        raise ValueError("requested query block subset is empty")

    scale_num, scale_den = _ratio_template(scale)
    output_shape = (batch, head_count, tokens_out, head_dim)
    output = forward_kernel(
        inputs=[q, k, v, preparation.q_mean, preparation.k_mean, preparation.v_sum, preparation.thresholds],
        template=[
            ("T", q.dtype),
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("NB", block_count),
            ("BLOCK", block_size),
            ("HEAD_START", head_start),
            ("HEAD_COUNT", head_count),
            ("Q_BLOCK_START", q_block_start),
            ("Q_BLOCK_COUNT", q_block_count),
            ("Q_TOKEN_START", q_token_start),
            ("TOKENS_OUT", tokens_out),
            ("PREFIX_QUERY_TOKENS", int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0)),
            ("SINK_BLOCK_START", int(rounded_sink_block_range[0])),
            ("SINK_BLOCK_STOP", int(rounded_sink_block_range[1])),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
        ],
        grid=(256, batch * head_count * q_block_count, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    return MacSolMetalAttentionResult(
        output=output,
        preparation=preparation,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        q_block_count=q_block_count,
        q_token_range=(q_token_start, q_token_stop),
        rounded_sink_block_range=(int(rounded_sink_block_range[0]), int(rounded_sink_block_range[1])),
        prefix_query_tokens=int(config.prefix_query_tokens if config.force_prefix_queries_dense else 0),
    )


def macsol_exact_tile_metal_v5(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    q_block_start: int = 0,
    kv_block_start: int = 0,
    kv_block_count: int = 1,
    head_start: int = 0,
    head_count: int | None = None,
    block_size: int = 64,
    scale: float | None = None,
) -> MacSolMetalExactTileResult:
    """Run the v5 64x64 exact-attention tile microkernel on full blocks.

    This is a bounded microbench/prototype, not a full MacSol sparse forward: it computes exact
    attention for one query block against one or more contiguous KV blocks independently and returns
    per-tile outputs with shape ``[B, head_count, kv_block_count, block_size, D]``.  It exists to
    test whether a threadgroup-cooperative 64x64 BF16 tile is a viable building block before any
    sparse online-softmax integration is attempted.
    """

    batch, heads, sequence, head_dim = _validate_qkv(q, k, v)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size != 64:
        raise ValueError(f"MacSol Metal v5 exact-tile microkernel is intentionally limited to 64-token blocks, got {block_size}")
    if int(head_dim) <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    q_block_start = int(q_block_start)
    kv_block_start = int(kv_block_start)
    kv_block_count = int(kv_block_count)
    head_start = int(head_start)
    if head_start < 0 or head_start >= heads:
        raise ValueError(f"head_start must be in [0,{heads}), got {head_start}")
    if head_count is None:
        head_count = heads - head_start
    head_count = int(head_count)
    if head_count <= 0 or head_start + head_count > heads:
        raise ValueError(f"head_count must keep range inside [0,{heads}], got start={head_start} count={head_count}")
    if q_block_start < 0 or q_block_start * block_size + block_size > sequence:
        raise ValueError(
            f"q_block_start must identify a full {block_size}-token block inside sequence {sequence}, got {q_block_start}"
        )
    if kv_block_start < 0 or kv_block_count <= 0 or (kv_block_start + kv_block_count) * block_size > sequence:
        raise ValueError(
            f"kv_block_start/count must identify full {block_size}-token blocks inside sequence {sequence}, "
            f"got start={kv_block_start} count={kv_block_count}"
        )
    scale = float(head_dim ** -0.5 if scale is None else scale)
    forward_kernel = _exact_tile_kernel_v5()
    if forward_kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")

    scale_num, scale_den = _ratio_template(scale)
    output_shape = (batch, head_count, kv_block_count, block_size, head_dim)
    output = forward_kernel(
        inputs=[q, k, v],
        template=[
            ("T", q.dtype),
            ("S", sequence),
            ("H", heads),
            ("D", head_dim),
            ("BLOCK", block_size),
            ("THREADS", 256),
            ("HEAD_START", head_start),
            ("HEAD_COUNT", head_count),
            ("Q_BLOCK_START", q_block_start),
            ("KV_BLOCK_START", kv_block_start),
            ("KV_BLOCK_COUNT", kv_block_count),
            ("SCALE_NUM", scale_num),
            ("SCALE_DEN", scale_den),
        ],
        grid=(256, batch * head_count * kv_block_count, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )[0]
    return MacSolMetalExactTileResult(
        output=output,
        head_start=head_start,
        head_count=head_count,
        q_block_start=q_block_start,
        kv_block_start=kv_block_start,
        kv_block_count=kv_block_count,
        q_token_range=(q_block_start * block_size, q_block_start * block_size + block_size),
        kv_token_range=(kv_block_start * block_size, (kv_block_start + kv_block_count) * block_size),
        block_size=block_size,
        scale=scale,
    )


__all__ = [
    "MACSOL_METAL_REJECTION_REASON",
    "MACSOL_METAL_STATUS",
    "MACSOL_METAL_V2_DESCRIPTION",
    "MACSOL_METAL_V2_REJECTION_REASON",
    "MACSOL_METAL_V2_STATUS",
    "MACSOL_METAL_V3_DESCRIPTION",
    "MACSOL_METAL_V3_REJECTION_REASON",
    "MACSOL_METAL_V3_STATUS",
    "MACSOL_METAL_V4_DESCRIPTION",
    "MACSOL_METAL_V4_REJECTION_REASON",
    "MACSOL_METAL_V4_STATUS",
    "MACSOL_METAL_V5_DECISION_NOTE",
    "MACSOL_METAL_V5_DESCRIPTION",
    "MACSOL_METAL_V5_REJECTION_REASON",
    "MACSOL_METAL_V5_STATUS",
    "MacSolMetalAttentionResult",
    "MacSolMetalExactTileResult",
    "MacSolMetalPreparation",
    "has_macsol_metal",
    "has_macsol_metal_v2",
    "has_macsol_metal_v3",
    "has_macsol_metal_v4",
    "has_macsol_metal_v5",
    "macsol_attention_metal_subset",
    "macsol_attention_metal_subset_v2",
    "macsol_attention_metal_subset_v3",
    "macsol_attention_metal_subset_v4",
    "macsol_exact_tile_metal_v5",
    "macsol_prepare_metal",
]
