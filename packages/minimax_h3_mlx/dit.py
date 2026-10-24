"""MLX port of ``MiniMaxH3DiTModel`` — the 33B MiniMax-H3 diffusion transformer.

The module tree reproduces the *original* checkpoint names, so the released safetensors load
1:1 with no weight surgery:

    video_patch_proj / audio_patch_proj / condition_proj
    time_embedder.proj_in / .proj_out
    token_refiner.blocks.{i}.{norm1,norm2,attn.*,mlp.*} + token_refiner.final_norm
    blocks.{i}.{norm1,norm2,attn.*,mlp.*,adaln_proj.linear}
    final_layer.{norm,adaln_proj.linear,video_out,audio_out}

Two raw-checkpoint layout quirks are handled by reshape in the forward pass rather than by
rewriting weights:

* ``attn.qkv_proj`` rows are **per-head interleaved** — ``[h0: q,k,v][h1: q,k,v]...`` — so the
  projection output reshapes to ``(..., heads, 3, head_dim)`` and indexes q/k/v on axis -2.
* ``mlp.fc1`` is a fused **``[gate; value]``** SwiGLU projection; the reference computes
  ``fc2(silu(gate) * value)``.

MiniMax-H3 runs one stack over a single packed 1-D sequence holding text, conditioning and
target video rows, and audio rows. Attention is full self-attention over that sequence; there
is no cross-attention and no per-modality block weights. Modality-specific behaviour comes only
from the two input patch projections, the per-row AdaLN modality tag, and the two output heads.
"""

from __future__ import annotations

from functools import lru_cache
import math

import mlx.core as mx
import mlx.nn as nn

from .activation_quant import project_linear
from .block_cache import BlockResidualCache
from .config import MODALITY_NUM, DiTConfig
from .forward_profile import profiled_call


DENSE_DEQUANT_PROFILE_OFF = "off"
DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED = "qkv-only-tiled"
DENSE_DEQUANT_PROFILE_FFN_FC2_TILED = "ffn-fc2-tiled"
DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT = "qkv-fc2-out-resident"
DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED = "qkv-fc2-out-tiled"
DENSE_DEQUANT_GENERATION_PROFILES = (
    DENSE_DEQUANT_PROFILE_OFF,
    DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
    DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
    DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
    DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
)


def normalize_dense_dequant_profile(profile: str | None) -> str:
    """Return the canonical disabled-by-default dense-dequant generation/profile name."""
    if profile is None:
        return DENSE_DEQUANT_PROFILE_OFF
    token = str(profile).strip().lower().replace("_", "-")
    aliases = {
        "": DENSE_DEQUANT_PROFILE_OFF,
        "none": DENSE_DEQUANT_PROFILE_OFF,
        "false": DENSE_DEQUANT_PROFILE_OFF,
        "0": DENSE_DEQUANT_PROFILE_OFF,
        DENSE_DEQUANT_PROFILE_OFF: DENSE_DEQUANT_PROFILE_OFF,
        "qkv": DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
        "qkv-tiled": DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
        "attention-qkv-tiled": DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
        DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED: DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
        "fc2": DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        "fc2-tiled": DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        "ffn-fc2": DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        "ffn-fc2-tile1024": DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        "ffn-fc2-tiled-dense-dequant": DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        "ffn-fc2-tiled-dense-dequant-tile1024": DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        DENSE_DEQUANT_PROFILE_FFN_FC2_TILED: DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        "resident": DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        "out-resident": DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        "out-dense": DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        "qkv-fc2-out-dense": DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT: DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        "tiled": DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
        "out-tiled": DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED: DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
    }
    try:
        return aliases[token]
    except KeyError as exc:
        allowed = ", ".join(DENSE_DEQUANT_GENERATION_PROFILES)
        raise ValueError(f"unknown dense-dequant profile {profile!r}; expected one of: {allowed}") from exc


def param_dtype(layer: nn.Module) -> mx.Dtype:
    """The dtype a layer's *activations* should be aligned to.

    The reference casts each input to its projection's parameter dtype, and reading
    ``layer.weight.dtype`` is the obvious way to do that — but it is wrong the moment the layer is
    quantized. ``QuantizedLinear.weight`` is **packed uint32 storage**, so casting activations to it
    truncates them to integers, silently and identically at every bit width. The scales carry the
    real compute dtype.
    """
    scales = getattr(layer, "scales", None)
    return scales.dtype if scales is not None else layer.weight.dtype


def linear_rank3_input_as_rank2(layer: nn.Module, x: mx.array) -> mx.array:
    """Run a ``[..., H]`` projection as explicit ``[B*S, H]`` when the input is rank-3.

    This is a disabled-by-default hotpath probe for MLX quantized ``nn.Linear`` QMM dispatch.
    It does not change weights, dtypes, row order, or mathematical semantics: only the leading
    ``[B, S]`` dimensions are flattened before the projection and restored afterwards.
    """
    if len(x.shape) != 3:
        return layer(x)
    batch, sequence, hidden = x.shape
    projected = layer(x.reshape(batch * sequence, hidden))
    return projected.reshape(batch, sequence, projected.shape[-1])


def linear_output_row_slice_projection(layer: nn.Module, x: mx.array, start: int, stop: int) -> mx.array:
    """Project only ``layer`` output rows ``[start:stop]`` without dense dequantization.

    For MLX ``QuantizedLinear`` this calls ``mx.quantized_matmul`` on sliced packed output rows,
    scales, and quantization biases.  For dense ``Linear`` it falls back to the equivalent dense
    row-sliced matmul.  The helper is intentionally local to opt-in scheduling probes and preserves
    the caller's leading dimensions, output row order, dtype, and optional learned bias slice.
    """
    start = int(start)
    stop = int(stop)
    if start < 0 or stop <= start:
        raise ValueError(f"invalid output-row slice [{start}:{stop}]")

    scales = getattr(layer, "scales", None)
    out_features = int(scales.shape[0]) if scales is not None else int(layer.weight.shape[0])
    if stop > out_features:
        raise ValueError(f"output-row slice [{start}:{stop}] exceeds layer output rows {out_features}")

    if scales is None:
        projected = x @ layer.weight[start:stop].T
    else:
        quantized_matmul = getattr(mx, "quantized_matmul", None)
        if quantized_matmul is None:
            raise RuntimeError("mx.quantized_matmul is unavailable; cannot run quantized output-row slice")
        biases = getattr(layer, "biases", None)
        projected = quantized_matmul(
            x,
            layer.weight[start:stop],
            scales=scales[start:stop],
            biases=biases[start:stop] if biases is not None else None,
            transpose=True,
            group_size=int(getattr(layer, "group_size")),
            bits=int(getattr(layer, "bits")),
            mode=str(getattr(layer, "mode", "affine")),
        )
    if "bias" in layer:
        projected = projected + layer.bias[start:stop]
    return projected


def qkv_headgroup_row_sliced_projection(
    layer: nn.Module,
    x: mx.array,
    heads: int,
    head_dim: int,
    heads_per_slice: int,
) -> mx.array:
    """Project MiniMax-H3 per-head-interleaved QKV rows in whole-head groups.

    ``attn.qkv_proj`` stores output rows as ``[h0:q,k,v][h1:q,k,v]...``.  This opt-in
    helper preserves that raw row order but issues one output-row-sliced projection per
    contiguous group of complete heads, then concatenates the partial QKV tensors.  For
    quantized ``nn.Linear`` it reuses ``linear_output_row_slice_projection`` so each group
    stays on ``mx.quantized_matmul`` with sliced packed rows/scales/biases rather than
    materializing a dense weight.
    """
    heads = int(heads)
    head_dim = int(head_dim)
    heads_per_slice = int(heads_per_slice)
    if heads <= 0 or head_dim <= 0:
        raise ValueError(f"invalid qkv head geometry: heads={heads}, head_dim={head_dim}")
    if heads_per_slice <= 0:
        raise ValueError(f"heads_per_slice must be positive, got {heads_per_slice}")

    rows_per_head = 3 * head_dim
    expected_out_features = heads * rows_per_head
    scales = getattr(layer, "scales", None)
    out_features = int(scales.shape[0]) if scales is not None else int(layer.weight.shape[0])
    if out_features != expected_out_features:
        raise ValueError(
            "qkv output rows do not match whole-head interleaved contract: "
            f"got {out_features}, expected {expected_out_features} "
            f"({heads} heads * 3 * head_dim {head_dim})"
        )

    heads_per_slice = min(heads_per_slice, heads)
    outputs: list[mx.array] = []
    for head_start in range(0, heads, heads_per_slice):
        head_stop = min(head_start + heads_per_slice, heads)
        row_start = head_start * rows_per_head
        row_stop = head_stop * rows_per_head
        outputs.append(linear_output_row_slice_projection(layer, x, row_start, row_stop))

    if len(outputs) == 1:
        return outputs[0]
    return mx.concatenate(outputs, axis=-1)


def quantized_matmul_input_chunked_projection(layer: nn.Module, x: mx.array, chunk_groups: int) -> mx.array:
    """Run a quantized projection by splitting the input feature/groups dimension.

    MLX stores quantized linear weights as packed output rows and input quantization groups.  This
    opt-in helper slices only complete input quantization groups, calls ``mx.quantized_matmul`` for
    each faithful packed-weight/scales/biases slice, accumulates the partial output vectors, and
    applies the learned ``nn.Linear`` bias exactly once after accumulation.  Dense layers fall back
    to their normal projection so accidentally enabling the probe on an unquantized tiny layer does
    not change default semantics.
    """
    chunk_groups = int(chunk_groups)
    if chunk_groups <= 0:
        raise ValueError(f"chunk_groups must be positive, got {chunk_groups}")

    scales = getattr(layer, "scales", None)
    if scales is None:
        return layer(x)

    quantized_matmul = getattr(mx, "quantized_matmul", None)
    if quantized_matmul is None:
        raise RuntimeError("mx.quantized_matmul is unavailable; cannot run quantized input chunks")

    group_size = int(getattr(layer, "group_size"))
    bits = int(getattr(layer, "bits"))
    if group_size <= 0 or bits <= 0:
        raise ValueError(f"invalid quantization metadata: group_size={group_size}, bits={bits}")
    total_groups = int(scales.shape[1])
    input_features = int(x.shape[-1])
    expected_input_features = total_groups * group_size
    if input_features != expected_input_features:
        raise ValueError(
            "input feature dimension must match quantized groups exactly for group-aligned chunks: "
            f"got {input_features}, expected {expected_input_features} "
            f"({total_groups} groups * group_size {group_size})"
        )

    packed_bits_per_group = group_size * bits
    if packed_bits_per_group % 32:
        raise ValueError(
            f"cannot slice packed quantized columns: group_size * bits = {packed_bits_per_group} is not uint32 aligned"
        )
    packed_cols_per_group = packed_bits_per_group // 32
    expected_packed_cols = total_groups * packed_cols_per_group
    if int(layer.weight.shape[1]) != expected_packed_cols:
        raise ValueError(
            "packed quantized weight shape is inconsistent with scales/groups: "
            f"weight columns {int(layer.weight.shape[1])}, expected {expected_packed_cols}"
        )

    biases = getattr(layer, "biases", None)
    chunk_groups = min(chunk_groups, total_groups)
    partials: list[mx.array] = []
    for group_start in range(0, total_groups, chunk_groups):
        group_stop = min(group_start + chunk_groups, total_groups)
        feature_start = group_start * group_size
        feature_stop = group_stop * group_size
        packed_start = group_start * packed_cols_per_group
        packed_stop = group_stop * packed_cols_per_group
        partials.append(
            quantized_matmul(
                x[..., feature_start:feature_stop],
                layer.weight[:, packed_start:packed_stop],
                scales=scales[:, group_start:group_stop],
                biases=biases[:, group_start:group_stop] if biases is not None else None,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=str(getattr(layer, "mode", "affine")),
            )
        )

    projected = partials[0]
    for partial in partials[1:]:
        projected = projected + partial
    if "bias" in layer:
        projected = projected + layer.bias
    return projected



def linear_input_range_partial_projection(layer: nn.Module, x: mx.array, start: int, stop: int) -> mx.array:
    """Project one input-feature range of ``layer`` without applying learned output bias.

    This is the FC2-side companion to :func:`linear_output_row_slice_projection`.  For MLX
    ``QuantizedLinear`` it slices complete input quantization groups in the packed weight columns
    and calls ``mx.quantized_matmul`` on that range.  The returned tensor is only the partial dot
    product for ``x[..., start:stop]``; callers that sum multiple ranges must add any learned
    ``nn.Linear`` bias exactly once after accumulation.
    """
    start = int(start)
    stop = int(stop)
    if start < 0 or stop <= start:
        raise ValueError(f"invalid input-feature slice [{start}:{stop}]")

    scales = getattr(layer, "scales", None)
    if scales is None:
        input_features = int(layer.weight.shape[1])
        if stop > input_features:
            raise ValueError(f"input-feature slice [{start}:{stop}] exceeds layer input features {input_features}")
        return x @ layer.weight[:, start:stop].T

    quantized_matmul = getattr(mx, "quantized_matmul", None)
    if quantized_matmul is None:
        raise RuntimeError("mx.quantized_matmul is unavailable; cannot run quantized input-feature slice")

    group_size = int(getattr(layer, "group_size"))
    bits = int(getattr(layer, "bits"))
    if group_size <= 0 or bits <= 0:
        raise ValueError(f"invalid quantization metadata: group_size={group_size}, bits={bits}")
    if start % group_size or stop % group_size:
        raise ValueError(
            "quantized input-feature slices must align to full quantization groups: "
            f"slice [{start}:{stop}], group_size={group_size}"
        )
    group_start = start // group_size
    group_stop = stop // group_size
    total_groups = int(scales.shape[1])
    if group_stop > total_groups:
        raise ValueError(
            f"input-feature slice groups [{group_start}:{group_stop}] exceed quantized groups {total_groups}"
        )

    packed_bits_per_group = group_size * bits
    if packed_bits_per_group % 32:
        raise ValueError(
            f"cannot slice packed quantized columns: group_size * bits = {packed_bits_per_group} is not uint32 aligned"
        )
    packed_cols_per_group = packed_bits_per_group // 32
    packed_start = group_start * packed_cols_per_group
    packed_stop = group_stop * packed_cols_per_group
    expected_packed_cols = total_groups * packed_cols_per_group
    if int(layer.weight.shape[1]) != expected_packed_cols:
        raise ValueError(
            "packed quantized weight shape is inconsistent with scales/groups: "
            f"weight columns {int(layer.weight.shape[1])}, expected {expected_packed_cols}"
        )

    biases = getattr(layer, "biases", None)
    return quantized_matmul(
        x,
        layer.weight[:, packed_start:packed_stop],
        scales=scales[:, group_start:group_stop],
        biases=biases[:, group_start:group_stop] if biases is not None else None,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=str(getattr(layer, "mode", "affine")),
    )



def dense_linear_projection(layer: nn.Module, x: mx.array, dense_weight: mx.array) -> mx.array:
    """Run ``layer`` with an already-materialized dense ``[out,in]`` weight matrix.

    This helper is intentionally small and local to opt-in probes: it preserves the same leading
    dimensions as ``nn.Linear``/``nn.QuantizedLinear`` and applies a stored bias if the layer has one.
    The caller owns any quantized-weight reconstruction and cache lifecycle.
    """
    projected = x @ dense_weight.T
    if "bias" in layer:
        projected = projected + layer.bias
    return projected


def tiled_dense_linear_projection(layer: nn.Module, x: mx.array, tile_size: int) -> mx.array:
    """Run ``layer`` by transiently dequantizing output-channel weight tiles.

    This opt-in helper targets quantized projections without keeping a full dense ``[out,in]``
    matrix resident.  It slices the layer's output rows, dequantizes one tile, evaluates that tile's
    dense projection, then drops the dense tile before moving to the next output-channel range.  The
    input is evaluated once up front so a lazy upstream graph is not replayed for every tile.  Output
    tiles are concatenated in ascending row order, preserving raw checkpoint row layout such as the
    per-head-interleaved ``attn.qkv_proj`` contract.
    """
    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    scales = getattr(layer, "scales", None)
    biases = getattr(layer, "biases", None)
    if scales is not None:
        dequantize = getattr(mx, "dequantize", None)
        if dequantize is None:
            raise RuntimeError("mx.dequantize is unavailable; cannot reconstruct quantized weight tiles")
        out_features = int(scales.shape[0])
    else:
        dequantize = None
        out_features = int(layer.weight.shape[0])

    if out_features <= 0:
        raise ValueError(f"layer has no output channels: {out_features}")

    # Avoid replaying the upstream fc1/SwiGLU lazy graph once per output tile.
    mx.eval(x)
    mx.synchronize()

    outputs: list[mx.array] = []
    for start in range(0, out_features, tile_size):
        stop = min(start + tile_size, out_features)
        if scales is None:
            dense_tile = layer.weight[start:stop]
        else:
            bias_tile = biases[start:stop] if biases is not None else None
            dense_tile = dequantize(
                layer.weight[start:stop],
                scales[start:stop],
                bias_tile,
                group_size=int(getattr(layer, "group_size")),
                bits=int(getattr(layer, "bits")),
                mode=str(getattr(layer, "mode", "affine")),
                dtype=scales.dtype,
            )
        projected = x @ dense_tile.T
        if "bias" in layer:
            projected = projected + layer.bias[start:stop]
        mx.eval(projected)
        mx.synchronize()
        outputs.append(projected)
        del dense_tile, projected

    if len(outputs) == 1:
        return outputs[0]
    return mx.concatenate(outputs, axis=-1)


@lru_cache(maxsize=1)
def _swiglu_activated_multiply_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_swiglu_activated_multiply",
        input_names=["activated_gate", "value"],
        output_names=["out"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            out[elem] = activated_gate[elem] * value[elem];
        """,
    )


def has_swiglu_from_fused_metal() -> bool:
    """Return whether the parity-safe SwiGLU Metal multiply probe is available."""
    try:
        return _swiglu_activated_multiply_metal_kernel() is not None
    except Exception:
        return False


def swiglu_activated_multiply_metal(activated_gate: mx.array, value: mx.array) -> mx.array:
    """Multiply native ``nn.silu(gate)`` by ``value`` with one custom Metal kernel.

    The original one-kernel Metal ``exp`` implementation was useful as a scheduling probe but did
    not match MLX's compiled ``nn.silu`` closely enough on real BF16 4-bit activations.  This helper
    keeps the native MLX activation for strict parity and probes only the final SwiGLU multiply
    boundary with custom Metal.
    """
    kernel = _swiglu_activated_multiply_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if activated_gate.shape != value.shape:
        raise ValueError(f"activated gate/value shape mismatch: {activated_gate.shape} vs {value.shape}")
    if activated_gate.dtype != value.dtype:
        raise ValueError(f"activated gate/value dtype mismatch: {activated_gate.dtype} vs {value.dtype}")
    return kernel(
        inputs=[activated_gate, value],
        template=[("T", activated_gate.dtype)],
        grid=(int(activated_gate.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[activated_gate.shape],
        output_dtypes=[activated_gate.dtype],
    )[0]


def swiglu_from_fused_metal(fused: mx.array, ffn: int) -> mx.array:
    """Compute ``silu(fused[..., :ffn]) * fused[..., ffn:]`` through the Metal multiply probe.

    This is an opt-in scheduling probe, not the default SwiGLU implementation.  It deliberately
    preserves native ``nn.silu`` semantics and uses custom Metal only for the final multiply, so the
    candidate can be judged by strict parity before any timing promotion decision.
    """
    if not fused.shape or int(fused.shape[-1]) != 2 * int(ffn):
        raise ValueError(f"expected fused last dimension {2 * int(ffn)}, got {fused.shape}")
    gate, value = fused[..., : int(ffn)], fused[..., int(ffn) :]
    return swiglu_activated_multiply_metal(nn.silu(gate), value)


def timestep_embedding(
    timesteps: mx.array,
    dim: int,
    max_period: float = 10000.0,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
) -> mx.array:
    """Sinusoidal timestep embedding matching diffusers ``Timesteps`` / ``get_timestep_embedding``.

    Timesteps are consumed unscaled in ``[0, 1]``.
    """
    half_dim = dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps.astype(mx.float32)[:, None] * emb[None, :]
    # diffusers builds [sin, cos] then swaps the halves when `flip_sin_to_cos`.
    if flip_sin_to_cos:
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedder(nn.Module):
    """The timestep MLP shared by every AdaLN projection (``proj_in`` -> silu -> ``proj_out``).

    Kept in float32: it is a float32 module in the released mixed-precision checkpoint, and every
    block reads the same ``temb``, so a rounding applied here biases every block's modulation
    identically at every sampling step and accumulates coherently over the denoising trajectory.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.proj_in = nn.Linear(config.timestep_input_dim, config.time_embed_hidden_size, bias=True)
        self.proj_out = nn.Linear(config.time_embed_hidden_size, config.time_embed_dim, bias=True)

    def __call__(self, sinusoid: mx.array) -> mx.array:
        return self.proj_out(nn.silu(self.proj_in(sinusoid)))


class RotaryPosEmbed3D:
    """3-axis rotary embedding over the ``(t, h, w)`` coordinates of the packed sequence.

    One ``inv_freq`` buffer of ``rope_inv_freq_len`` frequencies is shared by the three axes.
    Each axis contributes that many angles; the three blocks are concatenated to
    ``3 * rope_inv_freq_len`` and then concatenated with themselves, so the rotate-half convention
    rotates ``2 * 3 * rope_inv_freq_len`` of the head channels and passes the rest through.
    """

    def __init__(self, config: DiTConfig):
        n = config.rope_inv_freq_len
        self.inv_freq = 1.0 / (
            config.rope_theta ** (mx.arange(0, 2 * n, 2, dtype=mx.float32) / (2 * n))
        )

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (seq_len, 3) -> cos/sin: (seq_len, 2 * 3 * inv_freq_len)
        pos = position_ids.astype(mx.float32)
        freqs = pos[..., None] * self.inv_freq.reshape(1, 1, -1)  # (seq, 3, n)
        freqs = mx.concatenate([freqs[:, 0], freqs[:, 1], freqs[:, 2]], axis=-1)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(freqs), mx.sin(freqs)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the leading ``rotary_dim`` channels of every head, pass the rest through.

    ``x`` is ``(batch, heads, seq, head_dim)``; ``cos``/``sin`` are ``(seq, rotary_dim)``.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.astype(x.dtype)[None, None, :, :]
    sin = sin.astype(x.dtype)[None, None, :, :]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    out = x_rot * cos + rotated * sin
    if x_pass.shape[-1] == 0:
        return out
    return mx.concatenate([out, x_pass], axis=-1)


@lru_cache(maxsize=1)
def _rotary_qk_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_rotary_qk",
        input_names=["q", "k", "cos", "sin"],
        output_names=["q_out", "k_out"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint d = elem % D;
            uint seq = (elem / D) % S;

            if (d < R) {
                uint half_dim = R / 2;
                uint paired_elem = (d < half_dim) ? (elem + half_dim) : (elem - half_dim);
                T c = cos[seq * R + d];
                T s = sin[seq * R + d];
                T q_val = q[elem];
                T k_val = k[elem];
                T q_pair = q[paired_elem];
                T k_pair = k[paired_elem];
                if (d < half_dim) {
                    q_out[elem] = q_val * c - q_pair * s;
                    k_out[elem] = k_val * c - k_pair * s;
                } else {
                    q_out[elem] = q_val * c + q_pair * s;
                    k_out[elem] = k_val * c + k_pair * s;
                }
            } else {
                q_out[elem] = q[elem];
                k_out[elem] = k[elem];
            }
        """,
    )


def has_rotary_qk_metal() -> bool:
    """Return whether the local MLX build exposes the custom q/k RoPE Metal entry point."""
    try:
        return _rotary_qk_metal_kernel() is not None
    except Exception:
        return False


def apply_rotary_qk_metal(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    """Apply DiT RoPE to q/k with one opt-in custom Metal kernel.

    ``q``/``k`` are ``[B,H,S,D]`` and ``cos``/``sin`` are ``[S,R]``.  The helper is
    intentionally not used by default; it is a scheduling/materialization probe for the attention
    boundary before SDPA.  Cosine/sine inputs are cast to the q/k dtype first to mirror the baseline
    ``apply_rotary`` arithmetic contract.
    """
    kernel = _rotary_qk_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if q.shape != k.shape:
        raise ValueError(f"q/k shape mismatch: {q.shape} vs {k.shape}")
    if q.dtype != k.dtype:
        raise ValueError(f"q/k dtype mismatch: {q.dtype} vs {k.dtype}")
    if len(q.shape) != 4:
        raise ValueError(f"expected q/k rank 4 [B,H,S,D], got {q.shape}")
    if cos.shape != sin.shape or len(cos.shape) != 2:
        raise ValueError(f"expected matching rank-2 cos/sin, got {cos.shape} and {sin.shape}")
    _batch, _heads, seq, head_dim = q.shape
    rotary_dim = int(cos.shape[-1])
    if int(sin.shape[0]) != int(seq) or int(cos.shape[0]) != int(seq):
        raise ValueError(f"cos/sin sequence length must match q/k S={seq}, got {cos.shape} and {sin.shape}")
    if rotary_dim > int(head_dim):
        raise ValueError(f"rotary dim {rotary_dim} exceeds q/k head dim {head_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary dim must be even, got {rotary_dim}")

    cos_t = cos.astype(q.dtype)
    sin_t = sin.astype(q.dtype)
    return tuple(
        kernel(
            inputs=[q, k, cos_t, sin_t],
            template=[("T", q.dtype), ("S", int(seq)), ("D", int(head_dim)), ("R", rotary_dim)],
            grid=(int(q.size), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[q.shape, k.shape],
            output_dtypes=[q.dtype, k.dtype],
        )
    )


@lru_cache(maxsize=1)
def _qkv_rmsnorm_sdpa_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_qkv_rmsnorm_sdpa",
        input_names=["qkv", "q_weight", "k_weight"],
        output_names=["q_out", "k_out", "v_out"],
        source=r"""
            uint row = thread_position_in_grid.x;
            uint h = row % H;
            uint s = (row / H) % S;
            uint b = row / (H * S);
            uint qkv_base = (((b * S + s) * H + h) * 3 * D);
            uint out_base = (((b * H + h) * S + s) * D);

            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (uint d = 0; d < D; ++d) {
                float qv = static_cast<float>(qkv[qkv_base + d]);
                float kv = static_cast<float>(qkv[qkv_base + D + d]);
                q_sum += qv * qv;
                k_sum += kv * kv;
            }
            float eps = static_cast<float>(EPS_NUM) / static_cast<float>(EPS_DEN);
            float q_inv = metal::rsqrt(q_sum / static_cast<float>(D) + eps);
            float k_inv = metal::rsqrt(k_sum / static_cast<float>(D) + eps);

            for (uint d = 0; d < D; ++d) {
                float qv = static_cast<float>(qkv[qkv_base + d]);
                float kv = static_cast<float>(qkv[qkv_base + D + d]);
                q_out[out_base + d] = static_cast<T>(qv * q_inv * static_cast<float>(q_weight[d]));
                k_out[out_base + d] = static_cast<T>(kv * k_inv * static_cast<float>(k_weight[d]));
                v_out[out_base + d] = qkv[qkv_base + 2 * D + d];
            }
        """,
    )


def has_qkv_rmsnorm_sdpa_metal() -> bool:
    """Return whether the local MLX build exposes the fused q/k RMSNorm layout kernel."""
    try:
        return _qkv_rmsnorm_sdpa_metal_kernel() is not None
    except Exception:
        return False


def qkv_rmsnorm_sdpa_metal(
    qkv: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Materialize q/k/v in SDPA layout while applying q/k RMSNorm in one Metal launch.

    ``qkv`` is the raw per-head-interleaved projection reshaped as ``[B,S,H,3,D]``.  The helper
    writes ``q``, ``k`` and ``v`` directly as ``[B,H,S,D]``; q/k receive the same per-head RMSNorm
    weights as the MLX baseline while v is a pure layout materialization.  It is opt-in only and
    intentionally falls behind explicit parity/timing gates because the custom reduction may drift
    slightly from MLX's native RMSNorm implementation.
    """
    kernel = _qkv_rmsnorm_sdpa_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if len(qkv.shape) != 5:
        raise ValueError(f"expected qkv rank 5 [B,S,H,3,D], got {qkv.shape}")
    batch, seq, heads, three, head_dim = qkv.shape
    if int(three) != 3:
        raise ValueError(f"expected qkv shape [B,S,H,3,D], got {qkv.shape}")
    if tuple(q_weight.shape) != (int(head_dim),) or tuple(k_weight.shape) != (int(head_dim),):
        raise ValueError(
            f"q/k RMSNorm weights must have shape ({int(head_dim)},), got {q_weight.shape} and {k_weight.shape}"
        )

    q_shape = (int(batch), int(heads), int(seq), int(head_dim))
    eps_den = 1_000_000_000
    eps_num = int(round(float(eps) * eps_den))
    return tuple(
        kernel(
            inputs=[qkv, q_weight, k_weight],
            template=[
                ("T", qkv.dtype),
                ("S", int(seq)),
                ("H", int(heads)),
                ("D", int(head_dim)),
                ("EPS_NUM", eps_num),
                ("EPS_DEN", eps_den),
            ],
            grid=(int(batch) * int(seq) * int(heads), 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[q_shape, q_shape, q_shape],
            output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype],
        )
    )


@lru_cache(maxsize=1)
def _qkv_rmsnorm_rotary_sdpa_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_qkv_rmsnorm_rotary_sdpa",
        input_names=["qkv", "q_weight", "k_weight", "cos", "sin"],
        output_names=["q_out", "k_out", "v_out"],
        source=r"""
            uint row = thread_position_in_grid.x;
            uint h = row % H;
            uint s = (row / H) % S;
            uint b = row / (H * S);
            uint qkv_base = (((b * S + s) * H + h) * 3 * D);
            uint out_base = (((b * H + h) * S + s) * D);

            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (uint d = 0; d < D; ++d) {
                float qv = static_cast<float>(qkv[qkv_base + d]);
                float kv = static_cast<float>(qkv[qkv_base + D + d]);
                q_sum += qv * qv;
                k_sum += kv * kv;
            }
            float eps = static_cast<float>(EPS_NUM) / static_cast<float>(EPS_DEN);
            float q_inv = metal::rsqrt(q_sum / static_cast<float>(D) + eps);
            float k_inv = metal::rsqrt(k_sum / static_cast<float>(D) + eps);
            uint half_rotary = R / 2;

            for (uint d = 0; d < D; ++d) {
                uint paired_d = d < half_rotary ? d + half_rotary : d - half_rotary;
                // mx.fast.rms_norm rounds the unit-normalized BF16 activation
                // before multiplying by the BF16 weight. Preserve that staged
                // rounding before the RoPE arithmetic; the prior fused chain
                // normalized and weighted in one float32 expression and failed
                // real-block parity.
                T q_unit = static_cast<T>(static_cast<float>(qkv[qkv_base + d]) * q_inv);
                T k_unit = static_cast<T>(static_cast<float>(qkv[qkv_base + D + d]) * k_inv);
                T q_norm = static_cast<T>(q_unit * q_weight[d]);
                T k_norm = static_cast<T>(k_unit * k_weight[d]);

                if (d < R) {
                    T q_pair_unit = static_cast<T>(static_cast<float>(qkv[qkv_base + paired_d]) * q_inv);
                    T k_pair_unit = static_cast<T>(static_cast<float>(qkv[qkv_base + D + paired_d]) * k_inv);
                    T q_pair = static_cast<T>(q_pair_unit * q_weight[paired_d]);
                    T k_pair = static_cast<T>(k_pair_unit * k_weight[paired_d]);
                    T c = cos[s * R + d];
                    T sv = sin[s * R + d];
                    T q_main = static_cast<T>(q_norm * c);
                    T k_main = static_cast<T>(k_norm * c);
                    T q_rot = static_cast<T>(q_pair * sv);
                    T k_rot = static_cast<T>(k_pair * sv);
                    if (d < half_rotary) {
                        q_out[out_base + d] = static_cast<T>(q_main - q_rot);
                        k_out[out_base + d] = static_cast<T>(k_main - k_rot);
                    } else {
                        q_out[out_base + d] = static_cast<T>(q_main + q_rot);
                        k_out[out_base + d] = static_cast<T>(k_main + k_rot);
                    }
                } else {
                    q_out[out_base + d] = q_norm;
                    k_out[out_base + d] = k_norm;
                }
                v_out[out_base + d] = qkv[qkv_base + 2 * D + d];
            }
        """,
    )


def has_qkv_rmsnorm_rotary_sdpa_metal() -> bool:
    """Return whether the local MLX build exposes the fused q/k RMSNorm+RoPE layout kernel."""
    try:
        return _qkv_rmsnorm_rotary_sdpa_metal_kernel() is not None
    except Exception:
        return False


def qkv_rmsnorm_rotary_sdpa_metal(
    qkv: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    cos: mx.array,
    sin: mx.array,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Materialize RoPE-applied q/k/v in SDPA layout after q/k RMSNorm in one Metal launch.

    ``qkv`` is the raw per-head-interleaved projection reshaped as ``[B,S,H,3,D]`` and
    ``cos``/``sin`` are the DiT rotary tables ``[S,R]``. The helper writes SDPA-ready
    ``[B,H,S,D]`` q/k/v tensors: q/k receive per-head RMSNorm and RoPE, while v is a pure layout
    materialization. It is disabled by default because the fused reduction and BF16 arithmetic
    order must be accepted only by explicit parity and warm real-block timing gates.
    """
    kernel = _qkv_rmsnorm_rotary_sdpa_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if len(qkv.shape) != 5:
        raise ValueError(f"expected qkv rank 5 [B,S,H,3,D], got {qkv.shape}")
    batch, seq, heads, three, head_dim = qkv.shape
    if int(three) != 3:
        raise ValueError(f"expected qkv shape [B,S,H,3,D], got {qkv.shape}")
    if tuple(q_weight.shape) != (int(head_dim),) or tuple(k_weight.shape) != (int(head_dim),):
        raise ValueError(
            f"q/k RMSNorm weights must have shape ({int(head_dim)},), got {q_weight.shape} and {k_weight.shape}"
        )
    if cos.shape != sin.shape or len(cos.shape) != 2:
        raise ValueError(f"expected matching rank-2 cos/sin, got {cos.shape} and {sin.shape}")
    if int(cos.shape[0]) != int(seq):
        raise ValueError(f"cos/sin sequence length must match qkv S={seq}, got {cos.shape} and {sin.shape}")
    rotary_dim = int(cos.shape[-1])
    if rotary_dim > int(head_dim):
        raise ValueError(f"rotary dim {rotary_dim} exceeds head dim {head_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary dim must be even, got {rotary_dim}")

    q_shape = (int(batch), int(heads), int(seq), int(head_dim))
    eps_den = 1_000_000_000
    eps_num = int(round(float(eps) * eps_den))
    return tuple(
        kernel(
            inputs=[qkv, q_weight, k_weight, cos.astype(qkv.dtype), sin.astype(qkv.dtype)],
            template=[
                ("T", qkv.dtype),
                ("S", int(seq)),
                ("H", int(heads)),
                ("D", int(head_dim)),
                ("R", rotary_dim),
                ("EPS_NUM", eps_num),
                ("EPS_DEN", eps_den),
            ],
            grid=(int(batch) * int(seq) * int(heads), 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[q_shape, q_shape, q_shape],
            output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype],
        )
    )


@lru_cache(maxsize=1)
def _sdpa_out_layout_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_sdpa_out_to_bshd",
        input_names=["sdpa_out"],
        output_names=["out"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint hd = elem % HD;
            uint s = (elem / HD) % S;
            uint b = elem / (S * HD);
            uint h = hd / D;
            uint d = hd - h * D;
            uint in_index = (((b * H + h) * S + s) * D + d);
            out[elem] = sdpa_out[in_index];
        """,
    )


def has_sdpa_out_layout_metal() -> bool:
    """Return whether the local MLX build exposes the SDPA-output layout copy kernel."""
    try:
        return _sdpa_out_layout_metal_kernel() is not None
    except Exception:
        return False


def sdpa_out_to_bshd_metal(sdpa_out: mx.array) -> mx.array:
    """Copy SDPA output from ``[B,H,S,D]`` directly to ``[B,S,H*D]`` with Metal.

    This helper is a disabled-by-default layout/materialization probe for the post-SDPA boundary.
    It performs no arithmetic: every output element reads exactly one element from the SDPA output
    tensor using the baseline transpose+reshape index mapping.  LoRA paths do not use this helper
    so adapter output projection deltas keep the existing baseline tensor path.
    """
    kernel = _sdpa_out_layout_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if len(sdpa_out.shape) != 4:
        raise ValueError(f"expected SDPA output rank 4 [B,H,S,D], got {sdpa_out.shape}")
    batch, heads, seq, head_dim = sdpa_out.shape
    out_shape = (int(batch), int(seq), int(heads) * int(head_dim))
    return kernel(
        inputs=[sdpa_out],
        template=[
            ("T", sdpa_out.dtype),
            ("S", int(seq)),
            ("H", int(heads)),
            ("D", int(head_dim)),
            ("HD", int(heads) * int(head_dim)),
        ],
        grid=(int(sdpa_out.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[sdpa_out.dtype],
    )[0]


def sdpa_head_batch_rank3(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array:
    """Call MLX SDPA with heads folded into the batch dimension.

    The default Attention path prepares q/k/v as ``[B,H,S,D]`` and calls rank-4
    ``mx.fast.scaled_dot_product_attention``.  This opt-in helper changes only that dispatch
    boundary: it reshapes matching q/k/v tensors to ``[B*H,S,D]`` for the SDPA call and reshapes the
    result back to ``[B,H,S,D]``.  It intentionally supports only the full-attention ``mask=None``
    path used by the MiniMax-H3 DiT block hotpath; masked or LoRA paths keep the baseline rank-4
    call.
    """
    if mask is not None:
        raise ValueError("sdpa_head_batch_rank3 only supports mask=None")
    if len(q.shape) != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected matching rank-4 q/k/v [B,H,S,D], got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"expected matching q/k/v dtypes, got {q.dtype}, {k.dtype}, {v.dtype}")
    batch, heads, seq, head_dim = q.shape
    rank3_shape = (int(batch) * int(heads), int(seq), int(head_dim))
    try:
        out = mx.fast.scaled_dot_product_attention(
            q.reshape(rank3_shape),
            k.reshape(rank3_shape),
            v.reshape(rank3_shape),
            scale=scale,
            mask=None,
        )
    except Exception as exc:
        raise RuntimeError(
            "mx.fast.scaled_dot_product_attention rejected rank-3 [B*H,S,D] q/k/v inputs"
        ) from exc
    return out.reshape(int(batch), int(heads), int(seq), int(head_dim))


def _sdpa_headgroup_mask(mask: mx.array | None, total_heads: int, head_start: int, head_stop: int) -> mx.array | None:
    """Return a mask view compatible with one contiguous rank-4 SDPA head group."""
    if mask is None:
        return None
    # MLX SDPA masks are broadcast against attention scores whose last three axes are
    # ``[H, query, key]``.  Broadcast masks (for example ``[S,S]`` or ``[B,1,S,S]``) can be
    # reused as-is, while masks with an explicit per-head axis must be sliced with q/k/v.
    if len(mask.shape) >= 3 and int(mask.shape[-3]) == int(total_heads):
        index = [slice(None)] * len(mask.shape)
        index[-3] = slice(int(head_start), int(head_stop))
        return mask[tuple(index)]
    return mask


def sdpa_headgroup_split_rank4(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: mx.array | None = None,
    heads_per_group: int = 8,
) -> mx.array:
    """Call rank-4 MLX SDPA separately on contiguous head groups and concatenate.

    This disabled-by-default scheduling probe keeps q/k/v in the supported ``[B,H,S,D]``
    layout.  It changes only the SDPA dispatch granularity: each call receives a contiguous
    slice of heads, the same scale, and an equivalent mask view.  Attention is independent across
    heads, so concatenating the per-group outputs along the head axis restores the baseline layout.
    """
    if len(q.shape) != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected matching rank-4 q/k/v [B,H,S,D], got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"expected matching q/k/v dtypes, got {q.dtype}, {k.dtype}, {v.dtype}")
    heads_per_group = int(heads_per_group)
    if heads_per_group <= 0:
        raise ValueError(f"heads_per_group must be positive, got {heads_per_group}")

    batch, heads, seq, head_dim = q.shape
    del batch, seq, head_dim
    total_heads = int(heads)
    if total_heads <= 0:
        raise ValueError(f"expected at least one attention head, got {total_heads}")

    group_heads = min(heads_per_group, total_heads)
    outputs: list[mx.array] = []
    for head_start in range(0, total_heads, group_heads):
        head_stop = min(head_start + group_heads, total_heads)
        group_mask = _sdpa_headgroup_mask(mask, total_heads, head_start, head_stop)
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, head_start:head_stop, :, :],
                k[:, head_start:head_stop, :, :],
                v[:, head_start:head_stop, :, :],
                scale=scale,
                mask=group_mask,
            )
        )
    if len(outputs) == 1:
        return outputs[0]
    return mx.concatenate(outputs, axis=1)


def materialize_sdpa_inputs_contiguous(
    q: mx.array,
    k: mx.array,
    v: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Force q/k/v into contiguous ``[B,H,S,D]`` buffers immediately before SDPA.

    This is a disabled-by-default pure layout-materialization probe. It performs no arithmetic and
    preserves shape, dtype, and element order; the only requested change is making MLX realize each
    SDPA input through ``mx.contiguous`` just before ``scaled_dot_product_attention``.
    """
    if len(q.shape) != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected matching rank-4 q/k/v [B,H,S,D], got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"expected matching q/k/v dtypes, got {q.dtype}, {k.dtype}, {v.dtype}")
    return mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)


def materialize_attention_input_contiguous(x: mx.array, hidden_size: int | None = None) -> mx.array:
    """Force the AdaLN-normalized Attention input contiguous immediately before ``qkv_proj``.

    This disabled-by-default pure materialization probe targets the Attention QKV QMM boundary. It
    performs no arithmetic and preserves shape, dtype, and row order; the only requested change is
    making MLX realize the normalized/modulated rank-3 activation through ``mx.contiguous`` before
    the fused ``qkv_proj`` projection.
    """
    if len(x.shape) != 3:
        raise ValueError(f"expected Attention input rank-3 [B,S,H], got {x.shape}")
    if hidden_size is not None and int(x.shape[-1]) != int(hidden_size):
        raise ValueError(f"expected Attention input last dimension {int(hidden_size)}, got {x.shape}")
    return mx.contiguous(x)


def materialize_attention_output_contiguous(x: mx.array, inner_dim: int | None = None) -> mx.array:
    """Force merged SDPA output contiguous immediately before ``out_proj``.

    This disabled-by-default pure materialization probe targets the Attention output-projection QMM
    boundary. It performs no arithmetic and preserves shape, dtype, and row order; the only requested
    change is making MLX realize the merged-head rank-3 activation through ``mx.contiguous`` after
    ``scaled_dot_product_attention(...).transpose(...).reshape(...)`` and just before ``out_proj``.
    """
    if len(x.shape) != 3:
        raise ValueError(f"expected Attention output rank-3 [B,S,H*D], got {x.shape}")
    if inner_dim is not None and int(x.shape[-1]) != int(inner_dim):
        raise ValueError(f"expected Attention output last dimension {int(inner_dim)}, got {x.shape}")
    return mx.contiguous(x)


def materialize_ffn_input_contiguous(x: mx.array, hidden_size: int | None = None) -> mx.array:
    """Force the FFN activation input into a contiguous buffer immediately before ``fc1``.

    This is a disabled-by-default pure materialization probe for the FFN ``fc1`` QMM boundary. It
    performs no arithmetic and preserves shape, dtype, and element order; the only requested change
    is making MLX realize the normalized/modulated FFN input through ``mx.contiguous`` before the
    first FeedForward projection.
    """
    if len(x.shape) < 1:
        raise ValueError(f"expected FFN input tensor with a feature axis, got {x.shape}")
    if hidden_size is not None and int(x.shape[-1]) != int(hidden_size):
        raise ValueError(f"expected FFN input last dimension {int(hidden_size)}, got {x.shape}")
    return mx.contiguous(x)


def materialize_ffn_hidden_contiguous(hidden: mx.array, ffn: int | None = None) -> mx.array:
    """Force the SwiGLU hidden tensor into a contiguous buffer immediately before ``fc2``.

    This is a disabled-by-default pure materialization probe for the FFN ``fc2`` QMM boundary. It
    performs no arithmetic and preserves shape, dtype, and element order; the only requested change
    is making MLX realize ``silu(gate) * value`` through ``mx.contiguous`` before the second
    FeedForward projection.
    """
    if len(hidden.shape) < 1:
        raise ValueError(f"expected FFN hidden tensor with a feature axis, got {hidden.shape}")
    if ffn is not None and int(hidden.shape[-1]) != int(ffn):
        raise ValueError(f"expected FFN hidden last dimension {int(ffn)}, got {hidden.shape}")
    return mx.contiguous(hidden)


def gather_packed_modulation_rows(
    modulation: tuple[mx.array, ...],
    adaln_indices: mx.array,
) -> tuple[mx.array, ...]:
    """Gather all six AdaLN tensors through one packed row-index operation.

    The DiT block projection already produces the unique ``(timestep, modality)`` modulation
    table, not one projected row per packed token.  This helper is therefore only a disabled
    scheduling probe for the repeated per-token gather/materialization boundary: it stacks the six
    small modulation tables, gathers ``adaln_indices`` once from the packed table, and returns the
    same six ``[sequence, hidden]`` row tensors the baseline obtains with six independent gathers.
    """
    if len(modulation) != 6:
        raise ValueError(f"expected six AdaLN modulation tensors, got {len(modulation)}")
    reference_shape = tuple(modulation[0].shape)
    reference_dtype = modulation[0].dtype
    if len(reference_shape) != 2:
        raise ValueError(f"expected rank-2 modulation tensors [rows, hidden], got {reference_shape}")
    for index, tensor in enumerate(modulation[1:], start=1):
        if tuple(tensor.shape) != reference_shape:
            raise ValueError(
                f"modulation tensor {index} shape {tensor.shape} does not match {reference_shape}"
            )
        if tensor.dtype != reference_dtype:
            raise ValueError(
                f"modulation tensor {index} dtype {tensor.dtype} does not match {reference_dtype}"
            )
    packed = mx.stack(modulation, axis=0)
    gathered = packed[:, adaln_indices, :]
    return tuple(gathered[index] for index in range(6))


@lru_cache(maxsize=1)
def _indexed_adaln_affine_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_indexed_adaln_affine",
        input_names=["normed", "scale", "shift", "indices"],
        output_names=["out"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint hidden = elem % H;
            uint seq = (elem / H) % S;
            uint row = static_cast<uint>(indices[seq]);
            out[elem] = normed[elem] * (static_cast<T>(1.0f) + scale[row * H + hidden]) + shift[row * H + hidden];
        """,
    )


def has_indexed_adaln_affine_metal() -> bool:
    """Return whether the local MLX build exposes the indexed AdaLN affine Metal kernel."""
    try:
        return _indexed_adaln_affine_metal_kernel() is not None
    except Exception:
        return False


def indexed_adaln_affine_metal(
    normed: mx.array,
    scale: mx.array,
    shift: mx.array,
    adaln_indices: mx.array,
) -> mx.array:
    """Compute ``normed * (1 + scale[adaln_indices]) + shift[adaln_indices]`` in Metal.

    ``normed`` is a rank-3 ``[B,S,H]`` RMSNorm output, ``scale`` and ``shift`` are the AdaLN
    modulation tables ``[rows,H]``, and ``adaln_indices`` maps each sequence row to a table row.
    The helper is an opt-in scheduling probe for the two TransformerBlock modulation boundaries;
    it fuses the row gather plus affine pointwise arithmetic while leaving the default path and
    LoRA path unchanged.
    """
    kernel = _indexed_adaln_affine_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if len(normed.shape) != 3:
        raise ValueError(f"expected normed rank-3 [B,S,H], got {normed.shape}")
    if len(scale.shape) != 2 or scale.shape != shift.shape:
        raise ValueError(f"expected matching rank-2 scale/shift tables [rows,H], got {scale.shape} and {shift.shape}")
    batch, seq, hidden = normed.shape
    if int(scale.shape[-1]) != int(hidden):
        raise ValueError(f"scale/shift hidden dim {scale.shape[-1]} does not match normed hidden dim {hidden}")
    if tuple(adaln_indices.shape) != (int(seq),):
        raise ValueError(f"adaln_indices must have shape ({int(seq)},), got {adaln_indices.shape}")
    if normed.dtype != scale.dtype or normed.dtype != shift.dtype:
        raise ValueError(f"normed/scale/shift dtypes must match, got {normed.dtype}, {scale.dtype}, {shift.dtype}")
    if adaln_indices.dtype not in (mx.int32, mx.uint32):
        adaln_indices = adaln_indices.astype(mx.int32)
    return kernel(
        inputs=[normed, scale, shift, adaln_indices],
        template=[("T", normed.dtype), ("S", int(seq)), ("H", int(hidden))],
        grid=(int(normed.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[normed.shape],
        output_dtypes=[normed.dtype],
    )[0]


@lru_cache(maxsize=1)
def _indexed_gated_residual_metal_kernel():
    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        return None
    return metal_kernel(
        name="minimax_h3_indexed_gated_residual",
        input_names=["base", "gate", "indices", "branch"],
        output_names=["out"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint hidden = elem % H;
            uint seq = (elem / H) % S;
            uint gate_row = static_cast<uint>(indices[seq]);
            out[elem] = base[elem] + gate[gate_row * H + hidden] * branch[elem];
        """,
    )


def has_indexed_gated_residual_metal() -> bool:
    """Return whether the local MLX build exposes the indexed gated-residual Metal kernel."""
    try:
        return _indexed_gated_residual_metal_kernel() is not None
    except Exception:
        return False


def indexed_gated_residual_metal(
    base: mx.array,
    gate: mx.array,
    adaln_indices: mx.array,
    branch: mx.array,
) -> mx.array:
    """Compute ``base + gate[adaln_indices] * branch`` with one opt-in Metal kernel.

    ``base`` and ``branch`` are rank-3 ``[B,S,H]`` tensors, ``gate`` is the AdaLN table
    ``[rows,H]``, and ``adaln_indices`` is the per-sequence row index vector.  The helper fuses
    the gate gather, multiply, and residual add at the two DiT block residual boundaries.  It is
    intentionally disabled by default because BF16 arithmetic-order drift and kernel launch cost
    must be judged by explicit parity and warm timing.
    """
    kernel = _indexed_gated_residual_metal_kernel()
    if kernel is None:
        raise RuntimeError("mx.fast.metal_kernel is unavailable in this MLX build")
    if base.shape != branch.shape or len(base.shape) != 3:
        raise ValueError(f"expected matching rank-3 base/branch [B,S,H], got {base.shape} and {branch.shape}")
    if len(gate.shape) != 2:
        raise ValueError(f"expected rank-2 gate table [rows,H], got {gate.shape}")
    batch, seq, hidden = base.shape
    if int(gate.shape[-1]) != int(hidden):
        raise ValueError(f"gate hidden dim {gate.shape[-1]} does not match residual hidden dim {hidden}")
    if tuple(adaln_indices.shape) != (int(seq),):
        raise ValueError(f"adaln_indices must have shape ({int(seq)},), got {adaln_indices.shape}")
    if base.dtype != branch.dtype or base.dtype != gate.dtype:
        raise ValueError(f"base/gate/branch dtypes must match, got {base.dtype}, {gate.dtype}, {branch.dtype}")
    if adaln_indices.dtype not in (mx.int32, mx.uint32):
        adaln_indices = adaln_indices.astype(mx.int32)
    return kernel(
        inputs=[base, gate, adaln_indices, branch],
        template=[("T", base.dtype), ("S", int(seq)), ("H", int(hidden))],
        grid=(int(base.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[base.shape],
        output_dtypes=[base.dtype],
    )[0]


class Attention(nn.Module):
    """Full self-attention over the packed sequence, with per-head q/k RMSNorm."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self._hidden = config.hidden_size
        self._inner = config.inner_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.k_norm = nn.RMSNorm(config.attention_head_dim, eps=config.qk_norm_eps)
        self.qk_norm_eps = float(config.qk_norm_eps)
        self.out_proj = nn.Linear(config.inner_dim, config.hidden_size, bias=False)
        # Disabled-by-default strict-equivalent Attention hotpath probes. Benchmarks may enable
        # one selected candidate at a time; release/default behavior remains the original layout.
        self.use_pre_qkv_contiguous_candidate = False
        self.use_qkv_2d_projection_candidate = False
        self.use_out_2d_projection_candidate = False
        self.use_out_dense_dequant_candidate = False
        self.use_out_tiled_dense_dequant_candidate = False
        self.use_qkv_tiled_dense_dequant_candidate = False
        self.use_qkv_headgroup_row_sliced_qmm_candidate = False
        self.use_qkv_input_chunked_qmm_candidate = False
        self.use_qkv_pretranspose_layout_candidate = False
        self.use_qkv_rmsnorm_sdpa_metal_candidate = False
        self.use_qkv_rmsnorm_rotary_sdpa_metal_candidate = False
        self.use_rotary_qk_metal_candidate = False
        self.use_pre_sdpa_contiguous_candidate = False
        self.use_sdpa_head_batch_rank3_candidate = False
        self.use_sdpa_headgroup_split_candidate = False
        self.use_sdpa_out_layout_metal_candidate = False
        self.use_pre_out_proj_contiguous_candidate = False
        self.qkv_tiled_output_channels = 2048
        self.qkv_headgroup_heads_per_slice = 8
        self.qkv_input_chunk_groups = 42
        self.sdpa_headgroup_heads_per_group = 8
        self.out_tiled_output_channels = 2048
        self._out_dense_dequant_cache_key = None
        self._out_dense_dequant_cache = None

    def _pre_qkv_input(self, x: mx.array, lora=None) -> mx.array:
        if self.use_pre_qkv_contiguous_candidate and lora is None:
            return materialize_attention_input_contiguous(x, self._hidden)
        return x

    def _qkv_project(self, x: mx.array, lora=None) -> mx.array:
        x = self._pre_qkv_input(x, lora=lora)
        if self.use_qkv_headgroup_row_sliced_qmm_candidate and lora is None:
            return qkv_headgroup_row_sliced_projection(
                self.qkv_proj,
                x,
                self.heads,
                self.head_dim,
                self.qkv_headgroup_heads_per_slice,
            )
        if self.use_qkv_input_chunked_qmm_candidate and lora is None:
            return quantized_matmul_input_chunked_projection(self.qkv_proj, x, self.qkv_input_chunk_groups)
        if self.use_qkv_tiled_dense_dequant_candidate and lora is None:
            return tiled_dense_linear_projection(self.qkv_proj, x, self.qkv_tiled_output_channels)
        if self.use_qkv_2d_projection_candidate:
            return linear_rank3_input_as_rank2(self.qkv_proj, x)
        return project_linear(self.qkv_proj, x)

    def qkv_headgroup_row_sliced_qmm_info(self, heads_per_slice: int | None = None) -> dict[str, object]:
        """Return metadata for the whole-head-group row-sliced ``qkv_proj`` QMM probe."""
        scales = getattr(self.qkv_proj, "scales", None)
        biases = getattr(self.qkv_proj, "biases", None)
        source_is_quantized = scales is not None
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.qkv_proj.weight.shape[0])
        rows_per_head = int(3 * self.head_dim)
        expected_out_features = int(self.heads * rows_per_head)
        requested_heads = int(heads_per_slice if heads_per_slice is not None else self.qkv_headgroup_heads_per_slice)
        effective_heads = max(1, min(requested_heads, int(self.heads))) if requested_heads > 0 else requested_heads
        headgroup_count = int(math.ceil(int(self.heads) / effective_heads)) if effective_heads > 0 else 0
        tail_heads = ((int(self.heads) - 1) % effective_heads) + 1 if effective_heads > 0 else 0
        row_ranges = []
        if effective_heads > 0:
            for head_start in range(0, int(self.heads), effective_heads):
                head_stop = min(head_start + effective_heads, int(self.heads))
                row_ranges.append(
                    {
                        "head_range": [head_start, head_stop],
                        "row_range": [head_start * rows_per_head, head_stop * rows_per_head],
                    }
                )
        packed_source_nbytes = int(self.qkv_proj.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "uses_quantized_matmul": source_is_quantized,
            "dense_dequantization": False,
            "source_weight_shape": list(self.qkv_proj.weight.shape),
            "source_weight_dtype": str(self.qkv_proj.weight.dtype),
            "source_weight_nbytes": int(self.qkv_proj.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "input_features": int(self._hidden),
            "output_features": out_features,
            "expected_qkv_output_features": expected_out_features,
            "output_features_match_qkv_contract": out_features == expected_out_features,
            "attention_heads": int(self.heads),
            "head_dim": int(self.head_dim),
            "rows_per_head_group_unit": rows_per_head,
            "requested_heads_per_slice": requested_heads,
            "effective_heads_per_slice": effective_heads,
            "headgroup_count": headgroup_count,
            "tail_heads": tail_heads,
            "rows_per_full_slice": effective_heads * rows_per_head if effective_heads > 0 else None,
            "tail_rows": tail_heads * rows_per_head if effective_heads > 0 else None,
            "headgroup_row_ranges": row_ranges,
            "separate_projection_count": headgroup_count,
            "materializes_dense_weight": False,
            "materializes_full_qkv_before_concat": False,
            "row_order_contract": "each slice spans complete per-head-interleaved [q,k,v] row groups; concatenation restores original qkv_proj output row order",
            "helper": "qkv_headgroup_row_sliced_projection -> linear_output_row_slice_projection(mx.quantized_matmul on qkv rows [head_start*3D:head_stop*3D])",
            "group_size": int(getattr(self.qkv_proj, "group_size", 0) or 0),
            "bits": int(getattr(self.qkv_proj, "bits", 0) or 0),
            "mode": str(getattr(self.qkv_proj, "mode", "none")),
        }

    def qkv_input_chunked_qmm_info(self, chunk_groups: int | None = None) -> dict[str, object]:
        """Return metadata for the ``qkv_proj`` input/group chunked quantized-matmul probe."""
        scales = getattr(self.qkv_proj, "scales", None)
        biases = getattr(self.qkv_proj, "biases", None)
        source_is_quantized = scales is not None
        group_size = int(getattr(self.qkv_proj, "group_size", 0) or 0)
        bits = int(getattr(self.qkv_proj, "bits", 0) or 0)
        total_groups = int(scales.shape[1]) if scales is not None else 0
        input_features = total_groups * group_size if source_is_quantized else int(self._hidden)
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.qkv_proj.weight.shape[0])
        expected_out_features = int(self.heads * 3 * self.head_dim)
        requested_chunk_groups = int(chunk_groups if chunk_groups is not None else self.qkv_input_chunk_groups)
        effective_chunk_groups = max(1, min(requested_chunk_groups, total_groups)) if total_groups else max(1, requested_chunk_groups)
        chunk_count = int(math.ceil(total_groups / effective_chunk_groups)) if total_groups else 0
        tail_chunk_groups = ((total_groups - 1) % effective_chunk_groups) + 1 if total_groups else 0
        packed_cols_per_group = (group_size * bits // 32) if group_size and bits and (group_size * bits) % 32 == 0 else None
        packed_source_nbytes = int(self.qkv_proj.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "uses_quantized_matmul": source_is_quantized,
            "dense_dequantization": False,
            "source_weight_shape": list(self.qkv_proj.weight.shape),
            "source_weight_dtype": str(self.qkv_proj.weight.dtype),
            "source_weight_nbytes": int(self.qkv_proj.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "input_features": input_features,
            "output_features": out_features,
            "expected_qkv_output_features": expected_out_features,
            "output_features_match_qkv_contract": out_features == expected_out_features,
            "attention_heads": int(self.heads),
            "head_dim": int(self.head_dim),
            "group_size": group_size,
            "bits": bits,
            "mode": str(getattr(self.qkv_proj, "mode", "none")),
            "total_input_groups": total_groups,
            "requested_chunk_groups": requested_chunk_groups,
            "effective_chunk_groups": effective_chunk_groups,
            "chunk_count": chunk_count,
            "tail_chunk_groups": tail_chunk_groups,
            "features_per_full_chunk": effective_chunk_groups * group_size,
            "tail_features": tail_chunk_groups * group_size,
            "packed_cols_per_group": packed_cols_per_group,
            "packed_cols_per_full_chunk": (effective_chunk_groups * packed_cols_per_group) if packed_cols_per_group is not None else None,
            "learned_bias_present": "bias" in self.qkv_proj,
            "learned_bias_added_once_after_partial_accumulation": True,
            "partial_projection_count": chunk_count,
            "materializes_dense_weight": False,
            "materializes_full_qkv_before_accumulation": False,
            "row_order_contract": "partial QMMs cover input quantization groups only; output rows remain the original per-head-interleaved [h0:q,k,v][h1:q,k,v]... order",
            "helper": "quantized_matmul_input_chunked_projection(mx.quantized_matmul over qkv_proj input quantization-group chunks)",
        }

    def qkv_tiled_dense_dequant_info(self, tile_size: int | None = None) -> dict[str, object]:
        """Return metadata for the transient tiled dense-``qkv_proj`` opt-in path."""
        scales = getattr(self.qkv_proj, "scales", None)
        biases = getattr(self.qkv_proj, "biases", None)
        source_is_quantized = scales is not None
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.qkv_proj.weight.shape[0])
        input_features = int(self._hidden)
        requested_tile = int(tile_size if tile_size is not None else self.qkv_tiled_output_channels)
        tile_rows = max(1, min(requested_tile, out_features)) if out_features else max(1, requested_tile)
        tile_count = int(math.ceil(out_features / tile_rows)) if out_features else 0
        tail_rows = ((out_features - 1) % tile_rows) + 1 if out_features else 0
        dense_dtype = scales.dtype if source_is_quantized else self.qkv_proj.weight.dtype
        dense_dtype_nbytes = int(mx.zeros((1,), dtype=dense_dtype).nbytes)
        max_dense_tile_nbytes = int(tile_rows * input_features * dense_dtype_nbytes)
        full_dense_nbytes = int(out_features * input_features * dense_dtype_nbytes)
        packed_source_nbytes = int(self.qkv_proj.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "source_weight_shape": list(self.qkv_proj.weight.shape),
            "source_weight_dtype": str(self.qkv_proj.weight.dtype),
            "source_weight_nbytes": int(self.qkv_proj.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "output_features": out_features,
            "input_features": input_features,
            "requested_tile_output_channels": requested_tile,
            "effective_tile_output_channels": tile_rows,
            "tile_count": tile_count,
            "tail_tile_output_channels": tail_rows,
            "dense_tile_dtype": str(dense_dtype),
            "dense_tile_dtype_nbytes": dense_dtype_nbytes,
            "max_dense_tile_nbytes": max_dense_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "max_tile_fraction_of_full_dense": (max_dense_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "persistent_full_dense_allocation": False,
            "materialization_policy": "mx.eval input once, then mx.eval each output-channel tile before concatenation",
            "row_order_contract": "tiles are concatenated in original output-row order; qkv remains per-head interleaved [h0:q,k,v][h1:q,k,v]...",
            "group_size": int(getattr(self.qkv_proj, "group_size", 0) or 0),
            "bits": int(getattr(self.qkv_proj, "bits", 0) or 0),
            "mode": str(getattr(self.qkv_proj, "mode", "none")),
        }

    def out_tiled_dense_dequant_info(self, tile_size: int | None = None) -> dict[str, object]:
        """Return metadata for the transient tiled dense-``out_proj`` opt-in path."""
        scales = getattr(self.out_proj, "scales", None)
        biases = getattr(self.out_proj, "biases", None)
        source_is_quantized = scales is not None
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.out_proj.weight.shape[0])
        input_features = int(self._inner)
        requested_tile = int(tile_size if tile_size is not None else self.out_tiled_output_channels)
        tile_rows = max(1, min(requested_tile, out_features)) if out_features else max(1, requested_tile)
        tile_count = int(math.ceil(out_features / tile_rows)) if out_features else 0
        tail_rows = ((out_features - 1) % tile_rows) + 1 if out_features else 0
        dense_dtype = scales.dtype if source_is_quantized else self.out_proj.weight.dtype
        dense_dtype_nbytes = int(mx.zeros((1,), dtype=dense_dtype).nbytes)
        max_dense_tile_nbytes = int(tile_rows * input_features * dense_dtype_nbytes)
        full_dense_nbytes = int(out_features * input_features * dense_dtype_nbytes)
        packed_source_nbytes = int(self.out_proj.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "source_weight_shape": list(self.out_proj.weight.shape),
            "source_weight_dtype": str(self.out_proj.weight.dtype),
            "source_weight_nbytes": int(self.out_proj.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "output_features": out_features,
            "input_features": input_features,
            "requested_tile_output_channels": requested_tile,
            "effective_tile_output_channels": tile_rows,
            "tile_count": tile_count,
            "tail_tile_output_channels": tail_rows,
            "dense_tile_dtype": str(dense_dtype),
            "dense_tile_dtype_nbytes": dense_dtype_nbytes,
            "max_dense_tile_nbytes": max_dense_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "max_tile_fraction_of_full_dense": (max_dense_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "persistent_full_dense_allocation": False,
            "materialization_policy": "mx.eval input once, then mx.eval each out_proj output-channel tile before concatenation",
            "row_order_contract": "tiles are concatenated in original Attention.out_proj output-row order",
            "group_size": int(getattr(self.out_proj, "group_size", 0) or 0),
            "bits": int(getattr(self.out_proj, "bits", 0) or 0),
            "mode": str(getattr(self.out_proj, "mode", "none")),
        }

    def clear_out_dense_dequant_cache(self) -> None:
        """Drop any resident dense ``out_proj`` weight reconstructed by the opt-in probe."""
        self._out_dense_dequant_cache_key = None
        self._out_dense_dequant_cache = None

    def _out_dense_dequant_source_key(self):
        scales = getattr(self.out_proj, "scales", None)
        if scales is None:
            return None
        biases = getattr(self.out_proj, "biases", None)
        return (
            id(self.out_proj.weight),
            id(scales),
            id(biases),
            tuple(self.out_proj.weight.shape),
            tuple(scales.shape),
            tuple(biases.shape) if biases is not None else None,
            str(self.out_proj.weight.dtype),
            str(scales.dtype),
            str(biases.dtype) if biases is not None else None,
            int(getattr(self.out_proj, "group_size")),
            int(getattr(self.out_proj, "bits")),
            str(getattr(self.out_proj, "mode", "affine")),
        )

    def _out_dense_dequant_weight(self) -> mx.array:
        scales = getattr(self.out_proj, "scales", None)
        if scales is None:
            return self.out_proj.weight
        key = self._out_dense_dequant_source_key()
        dense = self._out_dense_dequant_cache if self._out_dense_dequant_cache_key == key else None
        if dense is None:
            dequantize = getattr(mx, "dequantize", None)
            if dequantize is None:
                raise RuntimeError("mx.dequantize is unavailable; cannot reconstruct quantized out_proj weight")
            dense = dequantize(
                self.out_proj.weight,
                scales,
                getattr(self.out_proj, "biases", None),
                group_size=int(getattr(self.out_proj, "group_size")),
                bits=int(getattr(self.out_proj, "bits")),
                mode=str(getattr(self.out_proj, "mode", "affine")),
                dtype=scales.dtype,
            )
            self._out_dense_dequant_cache_key = key
            self._out_dense_dequant_cache = dense
        return dense

    def out_dense_dequant_cache_info(self) -> dict[str, object]:
        """Return small metadata for the resident dense-``out_proj`` opt-in cache."""
        dense = self._out_dense_dequant_cache
        scales = getattr(self.out_proj, "scales", None)
        biases = getattr(self.out_proj, "biases", None)
        return {
            "source_is_quantized": scales is not None,
            "cached": dense is not None,
            "dense_shape": list(dense.shape) if dense is not None else None,
            "dense_dtype": str(dense.dtype) if dense is not None else None,
            "dense_nbytes": int(dense.nbytes) if dense is not None else 0,
            "source_weight_shape": list(self.out_proj.weight.shape),
            "source_weight_dtype": str(self.out_proj.weight.dtype),
            "source_weight_nbytes": int(self.out_proj.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "group_size": int(getattr(self.out_proj, "group_size", 0) or 0),
            "bits": int(getattr(self.out_proj, "bits", 0) or 0),
            "mode": str(getattr(self.out_proj, "mode", "none")),
        }

    def _pre_out_project_input(self, x: mx.array, lora=None) -> mx.array:
        if self.use_pre_out_proj_contiguous_candidate and lora is None:
            return materialize_attention_output_contiguous(x, self._inner)
        return x

    def _out_project(self, x: mx.array, lora=None) -> mx.array:
        x = self._pre_out_project_input(x, lora=lora)
        if self.use_out_dense_dequant_candidate and lora is None:
            return dense_linear_projection(self.out_proj, x, self._out_dense_dequant_weight())
        if self.use_out_tiled_dense_dequant_candidate and lora is None:
            return tiled_dense_linear_projection(self.out_proj, x, self.out_tiled_output_channels)
        if self.use_out_2d_projection_candidate and lora is None:
            return linear_rank3_input_as_rank2(self.out_proj, x)
        return project_linear(self.out_proj, x)

    def _qkv_sdpa_tensors(self, x: mx.array, lora=None) -> tuple[mx.array, mx.array, mx.array]:
        """Project QKV and return ``q, k, v`` in ``[B, H, S, D]`` SDPA layout.

        The opt-in pretranspose candidate changes only layout/materialization scheduling: after the
        raw per-head-interleaved projection is reshaped to ``[B, S, H, 3, D]``, it transposes once to
        ``[B, H, 3, S, D]`` before slicing q/k/v.  With LoRA active the method deliberately falls
        back to the baseline order so additive adapter deltas keep their existing rank-3 path.
        """
        B, S, _ = x.shape
        qkv = profiled_call(
            "attention.qkv_projection",
            "linear_projection",
            lambda: self._qkv_project(x, lora=lora).reshape(B, S, self.heads, 3, self.head_dim),
            metadata={"heads": self.heads, "head_dim": self.head_dim, "sequence_length": S},
        )
        if lora is None:
            if self.use_qkv_rmsnorm_sdpa_metal_candidate:
                return profiled_call(
                    "attention.qkv_rmsnorm_sdpa_layout_metal",
                    "attention_sdpa",
                    lambda: qkv_rmsnorm_sdpa_metal(qkv, self.q_norm.weight, self.k_norm.weight, self.qk_norm_eps),
                    metadata={"heads": self.heads, "head_dim": self.head_dim, "sequence_length": S},
                )
            if self.use_qkv_pretranspose_layout_candidate:
                def pretranspose() -> tuple[mx.array, mx.array, mx.array]:
                    qkv_t = qkv.transpose(0, 2, 3, 1, 4)
                    q = self.q_norm(qkv_t[:, :, 0])
                    k = self.k_norm(qkv_t[:, :, 1])
                    v = qkv_t[:, :, 2]
                    return q, k, v

                return profiled_call(
                    "attention.qk_norm_v_layout",
                    "attention_sdpa",
                    pretranspose,
                    metadata={"variant": "pretranspose", "sequence_length": S},
                )

        # Raw-checkpoint QKV rows are per-head interleaved: (..., heads, 3, head_dim).
        def qk_norm_v_layout() -> tuple[mx.array, mx.array, mx.array]:
            if lora is not None and lora.has("attn.qkv_proj"):
                native_delta = lora.apply("attn.qkv_proj", x).reshape(
                    B, S, self.heads, 3, self.head_dim
                )
                qkv_with_lora = qkv + native_delta
            else:
                qkv_with_lora = qkv
            q, k, v = (
                qkv_with_lora[:, :, :, 0],
                qkv_with_lora[:, :, :, 1],
                qkv_with_lora[:, :, :, 2],
            )
            if lora is not None and not lora.has("attn.qkv_proj"):
                q = q + lora.apply("attn.to_q", x).reshape(B, S, self.heads, self.head_dim)
                k = k + lora.apply("attn.to_k", x).reshape(B, S, self.heads, self.head_dim)
                v = v + lora.apply("attn.to_v", x).reshape(B, S, self.heads, self.head_dim)
            q = self.q_norm(q).transpose(0, 2, 1, 3)
            k = self.k_norm(k).transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            return q, k, v

        return profiled_call(
            "attention.qk_norm_v_layout",
            "attention_sdpa",
            qk_norm_v_layout,
            metadata={"variant": "baseline", "sequence_length": S},
        )

    def _qkv_rmsnorm_rotary_sdpa_tensors(
        self,
        x: mx.array,
        rotary: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Project qkv and return RoPE-applied q/k/v in SDPA layout via the fused Metal probe."""
        B, S, _ = x.shape
        qkv = self._qkv_project(x).reshape(B, S, self.heads, 3, self.head_dim)
        return qkv_rmsnorm_rotary_sdpa_metal(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            rotary[0],
            rotary[1],
            self.qk_norm_eps,
        )

    def _pre_sdpa_inputs(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        lora=None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if self.use_pre_sdpa_contiguous_candidate and lora is None:
            return materialize_sdpa_inputs_contiguous(q, k, v)
        return q, k, v

    def __call__(
        self,
        x: mx.array,
        rotary: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | None = None,
        lora=None,
    ) -> mx.array:
        B, S, _ = x.shape
        fused_rotary = False
        if rotary is not None and self.use_qkv_rmsnorm_rotary_sdpa_metal_candidate and lora is None:
            q, k, v = self._qkv_rmsnorm_rotary_sdpa_tensors(x, rotary)
            fused_rotary = True
        else:
            q, k, v = self._qkv_sdpa_tensors(x, lora=lora)

        if rotary is not None and not fused_rotary:
            if self.use_rotary_qk_metal_candidate and lora is None:
                q, k = profiled_call(
                    "attention.rotary_qk_metal",
                    "attention_sdpa",
                    lambda: apply_rotary_qk_metal(q, k, *rotary),
                    metadata={"sequence_length": S},
                )
            else:
                q, k = profiled_call(
                    "attention.rotary_qk",
                    "attention_sdpa",
                    lambda: (apply_rotary(q, *rotary), apply_rotary(k, *rotary)),
                    metadata={"sequence_length": S},
                )

        q, k, v = self._pre_sdpa_inputs(q, k, v, lora=lora)
        if self.use_sdpa_headgroup_split_candidate and lora is None:
            out = profiled_call(
                "attention.sdpa_headgroup_split",
                "attention_sdpa",
                lambda: sdpa_headgroup_split_rank4(
                    q,
                    k,
                    v,
                    scale=self.scale,
                    mask=mask,
                    heads_per_group=self.sdpa_headgroup_heads_per_group,
                ),
                metadata={"sequence_length": S, "heads": self.heads},
            )
        elif self.use_sdpa_head_batch_rank3_candidate and lora is None and mask is None:
            out = profiled_call(
                "attention.sdpa_head_batch_rank3",
                "attention_sdpa",
                lambda: sdpa_head_batch_rank3(q, k, v, scale=self.scale, mask=None),
                metadata={"sequence_length": S, "heads": self.heads},
            )
        else:
            out = profiled_call(
                "attention.sdpa",
                "attention_sdpa",
                lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask),
                metadata={"sequence_length": S, "heads": self.heads},
            )
        if self.use_sdpa_out_layout_metal_candidate and lora is None:
            out = profiled_call(
                "attention.sdpa_out_layout_metal",
                "attention_sdpa",
                lambda: sdpa_out_to_bshd_metal(out),
                metadata={"sequence_length": S},
            )
        else:
            out = profiled_call(
                "attention.sdpa_out_layout",
                "attention_sdpa",
                lambda: out.transpose(0, 2, 1, 3).reshape(B, S, self.heads * self.head_dim),
                metadata={"sequence_length": S},
            )
        projected = profiled_call(
            "attention.out_projection",
            "linear_projection",
            lambda: self._out_project(out.astype(x.dtype), lora=lora),
            metadata={"sequence_length": S, "input_features": self._inner, "output_features": self._hidden},
        )
        if lora is None:
            return projected
        target = "attn.out_proj" if lora.has("attn.out_proj") else "attn.to_out.0"
        return projected + lora.apply(target, out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward. ``fc1`` is the fused ``[gate; value]`` projection."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)
        self._ffn = config.ffn_hidden_size
        self._hidden = config.hidden_size
        # Disabled-by-default strict-equivalent hotpath probes for the dominant FFN boundary.
        # Benchmarks may opt into one selected candidate at a time; release/default behavior
        # remains the original rank-3 projection plus slice layout.
        self.use_mx_split_swiglu_candidate = False
        self.use_ffn_2d_projection_candidate = False
        self.use_ffn_fc1_rank2_qmm_candidate = False
        self.use_ffn_fc1_split_gate_value_quantized_qmm_candidate = False
        self.use_ffn_fc2_rank2_qmm_candidate = False
        self.use_ffn_fc1_dense_dequant_candidate = False
        self.use_ffn_fc1_tiled_dense_dequant_candidate = False
        self.use_ffn_fc2_dense_dequant_candidate = False
        self.use_ffn_fc2_tiled_dense_dequant_candidate = False
        self.use_ffn_fc2_input_chunked_qmm_candidate = False
        self.use_ffn_hidden_tile_stream_candidate = False
        self.use_ffn_metal_swiglu_candidate = False
        self.use_ffn_sequence_chunk_candidate = False
        self.use_ffn_pre_fc1_contiguous_candidate = False
        self.use_ffn_pre_fc2_contiguous_candidate = False
        self.use_ffn_subgraph_compile_candidate = False
        self.ffn_sequence_chunk_size = 512
        self.ffn_fc1_tiled_output_channels = 512
        self.ffn_fc2_tiled_output_channels = 512
        self.ffn_fc2_input_chunk_groups = 56
        self.ffn_hidden_tile_groups = 56
        self._fc1_dense_dequant_cache_key = None
        self._fc1_dense_dequant_cache = None
        self._fc2_dense_dequant_cache_key = None
        self._fc2_dense_dequant_cache = None
        self._ffn_subgraph_compile_cache_key = None
        self._ffn_subgraph_compile_cache = None

    def _pre_fc1_input(self, x: mx.array, lora=None) -> mx.array:
        if self.use_ffn_pre_fc1_contiguous_candidate and lora is None:
            return materialize_ffn_input_contiguous(x, self._hidden)
        return x

    def _fc1_gate_value_from_prepared_input(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return (
            linear_output_row_slice_projection(self.fc1, x, 0, self._ffn),
            linear_output_row_slice_projection(self.fc1, x, self._ffn, 2 * self._ffn),
        )

    def _fc1_split_gate_value_project(self, x: mx.array, lora=None) -> tuple[mx.array, mx.array]:
        if lora is not None:
            fused = self.fc1(self._pre_fc1_input(x, lora=lora))
            return fused[..., : self._ffn], fused[..., self._ffn :]
        return self._fc1_gate_value_from_prepared_input(self._pre_fc1_input(x, lora=lora))

    def _fc1_project(self, x: mx.array, lora=None) -> mx.array:
        x = self._pre_fc1_input(x, lora=lora)
        if self.use_ffn_fc1_split_gate_value_quantized_qmm_candidate and lora is None:
            gate, value = self._fc1_gate_value_from_prepared_input(x)
            return mx.concatenate([gate, value], axis=-1)
        if self.use_ffn_fc1_tiled_dense_dequant_candidate and lora is None:
            return tiled_dense_linear_projection(self.fc1, x, self.ffn_fc1_tiled_output_channels)
        if self.use_ffn_fc1_dense_dequant_candidate and lora is None:
            return dense_linear_projection(self.fc1, x, self._fc1_dense_dequant_weight())
        if (self.use_ffn_2d_projection_candidate or self.use_ffn_fc1_rank2_qmm_candidate) and lora is None:
            return linear_rank3_input_as_rank2(self.fc1, x)
        return project_linear(self.fc1, x)

    def fc1_split_gate_value_quantized_qmm_info(self) -> dict[str, object]:
        """Return metadata for the split gate/value ``fc1`` output-row QMM probe."""
        scales = getattr(self.fc1, "scales", None)
        biases = getattr(self.fc1, "biases", None)
        source_is_quantized = scales is not None
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.fc1.weight.shape[0])
        packed_source_nbytes = int(self.fc1.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "uses_quantized_matmul": source_is_quantized,
            "dense_dequantization": False,
            "source_weight_shape": list(self.fc1.weight.shape),
            "source_weight_dtype": str(self.fc1.weight.dtype),
            "source_weight_nbytes": int(self.fc1.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "input_features": int(self._hidden),
            "output_features": out_features,
            "ffn_hidden_features": int(self._ffn),
            "gate_output_rows": [0, int(self._ffn)],
            "value_output_rows": [int(self._ffn), int(2 * self._ffn)],
            "split_matches_fused_fc1_contract": out_features == int(2 * self._ffn),
            "separate_projection_count": 2,
            "materializes_fused_fc1_in_forward": False,
            "helper": "linear_output_row_slice_projection(mx.quantized_matmul on fc1 rows [gate] and [value])",
            "group_size": int(getattr(self.fc1, "group_size", 0) or 0),
            "bits": int(getattr(self.fc1, "bits", 0) or 0),
            "mode": str(getattr(self.fc1, "mode", "none")),
        }

    def clear_fc1_dense_dequant_cache(self) -> None:
        """Drop any resident dense ``fc1`` weight reconstructed by the opt-in probe."""
        self._fc1_dense_dequant_cache_key = None
        self._fc1_dense_dequant_cache = None

    def _fc1_dense_dequant_source_key(self):
        scales = getattr(self.fc1, "scales", None)
        if scales is None:
            return None
        biases = getattr(self.fc1, "biases", None)
        return (
            id(self.fc1.weight),
            id(scales),
            id(biases),
            tuple(self.fc1.weight.shape),
            tuple(scales.shape),
            tuple(biases.shape) if biases is not None else None,
            str(self.fc1.weight.dtype),
            str(scales.dtype),
            str(biases.dtype) if biases is not None else None,
            int(getattr(self.fc1, "group_size")),
            int(getattr(self.fc1, "bits")),
            str(getattr(self.fc1, "mode", "affine")),
        )

    def _fc1_dense_dequant_weight(self) -> mx.array:
        scales = getattr(self.fc1, "scales", None)
        if scales is None:
            return self.fc1.weight
        key = self._fc1_dense_dequant_source_key()
        dense = self._fc1_dense_dequant_cache if self._fc1_dense_dequant_cache_key == key else None
        if dense is None:
            dequantize = getattr(mx, "dequantize", None)
            if dequantize is None:
                raise RuntimeError("mx.dequantize is unavailable; cannot reconstruct quantized fc1 weight")
            dense = dequantize(
                self.fc1.weight,
                scales,
                getattr(self.fc1, "biases", None),
                group_size=int(getattr(self.fc1, "group_size")),
                bits=int(getattr(self.fc1, "bits")),
                mode=str(getattr(self.fc1, "mode", "affine")),
                dtype=scales.dtype,
            )
            self._fc1_dense_dequant_cache_key = key
            self._fc1_dense_dequant_cache = dense
        return dense

    def fc1_dense_dequant_cache_info(self) -> dict[str, object]:
        """Return small metadata for the resident dense-``fc1`` opt-in cache."""
        dense = self._fc1_dense_dequant_cache
        scales = getattr(self.fc1, "scales", None)
        biases = getattr(self.fc1, "biases", None)
        return {
            "source_is_quantized": scales is not None,
            "cached": dense is not None,
            "dense_shape": list(dense.shape) if dense is not None else None,
            "dense_dtype": str(dense.dtype) if dense is not None else None,
            "dense_nbytes": int(dense.nbytes) if dense is not None else 0,
            "source_weight_shape": list(self.fc1.weight.shape),
            "source_weight_dtype": str(self.fc1.weight.dtype),
            "source_weight_nbytes": int(self.fc1.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "group_size": int(getattr(self.fc1, "group_size", 0) or 0),
            "bits": int(getattr(self.fc1, "bits", 0) or 0),
            "mode": str(getattr(self.fc1, "mode", "none")),
        }

    def fc1_tiled_dense_dequant_info(self, tile_size: int | None = None) -> dict[str, object]:
        """Return metadata for the transient tiled dense-``fc1`` opt-in path."""
        scales = getattr(self.fc1, "scales", None)
        biases = getattr(self.fc1, "biases", None)
        source_is_quantized = scales is not None
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.fc1.weight.shape[0])
        input_features = int(self._hidden)
        requested_tile = int(tile_size if tile_size is not None else self.ffn_fc1_tiled_output_channels)
        tile_rows = max(1, min(requested_tile, out_features)) if out_features else max(1, requested_tile)
        tile_count = int(math.ceil(out_features / tile_rows)) if out_features else 0
        tail_rows = ((out_features - 1) % tile_rows) + 1 if out_features else 0
        dense_dtype = scales.dtype if source_is_quantized else self.fc1.weight.dtype
        dense_dtype_nbytes = int(mx.zeros((1,), dtype=dense_dtype).nbytes)
        max_dense_tile_nbytes = int(tile_rows * input_features * dense_dtype_nbytes)
        full_dense_nbytes = int(out_features * input_features * dense_dtype_nbytes)
        packed_source_nbytes = int(self.fc1.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "source_weight_shape": list(self.fc1.weight.shape),
            "source_weight_dtype": str(self.fc1.weight.dtype),
            "source_weight_nbytes": int(self.fc1.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "output_features": out_features,
            "input_features": input_features,
            "requested_tile_output_channels": requested_tile,
            "effective_tile_output_channels": tile_rows,
            "tile_count": tile_count,
            "tail_tile_output_channels": tail_rows,
            "dense_tile_dtype": str(dense_dtype),
            "dense_tile_dtype_nbytes": dense_dtype_nbytes,
            "max_dense_tile_nbytes": max_dense_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "max_tile_fraction_of_full_dense": (max_dense_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "persistent_full_dense_allocation": False,
            "materialization_policy": "mx.eval fc1 input once, then mx.eval each fused gate/value output-channel tile before concatenation",
            "fused_gate_value_row_order_preserved": True,
            "group_size": int(getattr(self.fc1, "group_size", 0) or 0),
            "bits": int(getattr(self.fc1, "bits", 0) or 0),
            "mode": str(getattr(self.fc1, "mode", "none")),
        }

    def _pre_fc2_hidden(self, hidden: mx.array, lora=None) -> mx.array:
        if self.use_ffn_pre_fc2_contiguous_candidate and lora is None:
            return materialize_ffn_hidden_contiguous(hidden, self._ffn)
        return hidden

    def clear_fc2_dense_dequant_cache(self) -> None:
        """Drop any resident dense ``fc2`` weight reconstructed by the opt-in probe."""
        self._fc2_dense_dequant_cache_key = None
        self._fc2_dense_dequant_cache = None

    def _fc2_dense_dequant_source_key(self):
        scales = getattr(self.fc2, "scales", None)
        if scales is None:
            return None
        biases = getattr(self.fc2, "biases", None)
        return (
            id(self.fc2.weight),
            id(scales),
            id(biases),
            tuple(self.fc2.weight.shape),
            tuple(scales.shape),
            tuple(biases.shape) if biases is not None else None,
            str(self.fc2.weight.dtype),
            str(scales.dtype),
            str(biases.dtype) if biases is not None else None,
            int(getattr(self.fc2, "group_size")),
            int(getattr(self.fc2, "bits")),
            str(getattr(self.fc2, "mode", "affine")),
        )

    def _fc2_dense_dequant_weight(self) -> mx.array:
        scales = getattr(self.fc2, "scales", None)
        if scales is None:
            return self.fc2.weight
        key = self._fc2_dense_dequant_source_key()
        dense = self._fc2_dense_dequant_cache if self._fc2_dense_dequant_cache_key == key else None
        if dense is None:
            dequantize = getattr(mx, "dequantize", None)
            if dequantize is None:
                raise RuntimeError("mx.dequantize is unavailable; cannot reconstruct quantized fc2 weight")
            dense = dequantize(
                self.fc2.weight,
                scales,
                getattr(self.fc2, "biases", None),
                group_size=int(getattr(self.fc2, "group_size")),
                bits=int(getattr(self.fc2, "bits")),
                mode=str(getattr(self.fc2, "mode", "affine")),
                dtype=scales.dtype,
            )
            self._fc2_dense_dequant_cache_key = key
            self._fc2_dense_dequant_cache = dense
        return dense

    def fc2_dense_dequant_cache_info(self) -> dict[str, object]:
        """Return small metadata for the resident dense-``fc2`` opt-in cache."""
        dense = self._fc2_dense_dequant_cache
        scales = getattr(self.fc2, "scales", None)
        biases = getattr(self.fc2, "biases", None)
        return {
            "source_is_quantized": scales is not None,
            "cached": dense is not None,
            "dense_shape": list(dense.shape) if dense is not None else None,
            "dense_dtype": str(dense.dtype) if dense is not None else None,
            "dense_nbytes": int(dense.nbytes) if dense is not None else 0,
            "source_weight_shape": list(self.fc2.weight.shape),
            "source_weight_dtype": str(self.fc2.weight.dtype),
            "source_weight_nbytes": int(self.fc2.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "group_size": int(getattr(self.fc2, "group_size", 0) or 0),
            "bits": int(getattr(self.fc2, "bits", 0) or 0),
            "mode": str(getattr(self.fc2, "mode", "none")),
        }

    def fc2_tiled_dense_dequant_info(self, tile_size: int | None = None) -> dict[str, object]:
        """Return metadata for the transient tiled dense-``fc2`` opt-in path."""
        scales = getattr(self.fc2, "scales", None)
        biases = getattr(self.fc2, "biases", None)
        source_is_quantized = scales is not None
        out_features = int(scales.shape[0]) if source_is_quantized else int(self.fc2.weight.shape[0])
        input_features = int(self._ffn)
        requested_tile = int(tile_size if tile_size is not None else self.ffn_fc2_tiled_output_channels)
        tile_rows = max(1, min(requested_tile, out_features)) if out_features else max(1, requested_tile)
        tile_count = int(math.ceil(out_features / tile_rows)) if out_features else 0
        tail_rows = ((out_features - 1) % tile_rows) + 1 if out_features else 0
        dense_dtype = scales.dtype if source_is_quantized else self.fc2.weight.dtype
        dense_dtype_nbytes = int(mx.zeros((1,), dtype=dense_dtype).nbytes)
        max_dense_tile_nbytes = int(tile_rows * input_features * dense_dtype_nbytes)
        full_dense_nbytes = int(out_features * input_features * dense_dtype_nbytes)
        packed_source_nbytes = int(self.fc2.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "source_weight_shape": list(self.fc2.weight.shape),
            "source_weight_dtype": str(self.fc2.weight.dtype),
            "source_weight_nbytes": int(self.fc2.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "output_features": out_features,
            "input_features": input_features,
            "requested_tile_output_channels": requested_tile,
            "effective_tile_output_channels": tile_rows,
            "tile_count": tile_count,
            "tail_tile_output_channels": tail_rows,
            "dense_tile_dtype": str(dense_dtype),
            "dense_tile_dtype_nbytes": dense_dtype_nbytes,
            "max_dense_tile_nbytes": max_dense_tile_nbytes,
            "full_dense_nbytes_if_resident": full_dense_nbytes,
            "max_tile_fraction_of_full_dense": (max_dense_tile_nbytes / full_dense_nbytes) if full_dense_nbytes else None,
            "persistent_full_dense_allocation": False,
            "materialization_policy": "mx.eval hidden once, then mx.eval each output-channel tile before concatenation",
            "group_size": int(getattr(self.fc2, "group_size", 0) or 0),
            "bits": int(getattr(self.fc2, "bits", 0) or 0),
            "mode": str(getattr(self.fc2, "mode", "none")),
        }

    def fc2_input_chunked_qmm_info(self, chunk_groups: int | None = None) -> dict[str, object]:
        """Return metadata for the ``fc2`` input/group chunked quantized-matmul probe."""
        scales = getattr(self.fc2, "scales", None)
        biases = getattr(self.fc2, "biases", None)
        source_is_quantized = scales is not None
        group_size = int(getattr(self.fc2, "group_size", 0) or 0)
        bits = int(getattr(self.fc2, "bits", 0) or 0)
        total_groups = int(scales.shape[1]) if scales is not None else 0
        input_features = total_groups * group_size if source_is_quantized else int(self._ffn)
        requested_chunk_groups = int(chunk_groups if chunk_groups is not None else self.ffn_fc2_input_chunk_groups)
        effective_chunk_groups = max(1, min(requested_chunk_groups, total_groups)) if total_groups else max(1, requested_chunk_groups)
        chunk_count = int(math.ceil(total_groups / effective_chunk_groups)) if total_groups else 0
        tail_chunk_groups = ((total_groups - 1) % effective_chunk_groups) + 1 if total_groups else 0
        packed_cols_per_group = (group_size * bits // 32) if group_size and bits and (group_size * bits) % 32 == 0 else None
        packed_source_nbytes = int(self.fc2.weight.nbytes)
        if scales is not None:
            packed_source_nbytes += int(scales.nbytes)
        if biases is not None:
            packed_source_nbytes += int(biases.nbytes)
        return {
            "source_is_quantized": source_is_quantized,
            "uses_quantized_matmul": source_is_quantized,
            "dense_dequantization": False,
            "source_weight_shape": list(self.fc2.weight.shape),
            "source_weight_dtype": str(self.fc2.weight.dtype),
            "source_weight_nbytes": int(self.fc2.weight.nbytes),
            "source_scales_shape": list(scales.shape) if scales is not None else None,
            "source_scales_dtype": str(scales.dtype) if scales is not None else None,
            "source_scales_nbytes": int(scales.nbytes) if scales is not None else 0,
            "source_biases_shape": list(biases.shape) if biases is not None else None,
            "source_biases_dtype": str(biases.dtype) if biases is not None else None,
            "source_biases_nbytes": int(biases.nbytes) if biases is not None else 0,
            "packed_quantized_source_nbytes": packed_source_nbytes,
            "input_features": input_features,
            "output_features": int(scales.shape[0]) if source_is_quantized else int(self.fc2.weight.shape[0]),
            "group_size": group_size,
            "bits": bits,
            "mode": str(getattr(self.fc2, "mode", "none")),
            "total_input_groups": total_groups,
            "requested_chunk_groups": requested_chunk_groups,
            "effective_chunk_groups": effective_chunk_groups,
            "chunk_count": chunk_count,
            "tail_chunk_groups": tail_chunk_groups,
            "features_per_full_chunk": effective_chunk_groups * group_size,
            "tail_features": tail_chunk_groups * group_size,
            "packed_cols_per_group": packed_cols_per_group,
            "packed_cols_per_full_chunk": (effective_chunk_groups * packed_cols_per_group) if packed_cols_per_group is not None else None,
            "learned_bias_present": "bias" in self.fc2,
            "learned_bias_added_once_after_partial_accumulation": True,
            "partial_projection_count": chunk_count,
            "materializes_dense_weight": False,
            "helper": "quantized_matmul_input_chunked_projection(mx.quantized_matmul over fc2 input quantization-group chunks)",
        }

    def hidden_tile_stream_info(
        self,
        tile_groups: int | None = None,
        *,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        activation_dtype_nbytes: int = 2,
    ) -> dict[str, object]:
        """Return metadata for the hidden-channel streamed FFN candidate.

        The candidate tiles the FFN hidden/FC2-input axis in whole quantization groups: each tile
        computes only the matching FC1 gate/value rows, applies SwiGLU, immediately projects that
        hidden tile through the matching FC2 packed input columns, and accumulates the output.
        """
        fc2_scales = getattr(self.fc2, "scales", None)
        source_is_quantized = fc2_scales is not None
        group_size = int(getattr(self.fc2, "group_size", 1) or 1) if source_is_quantized else 1
        total_groups = int(fc2_scales.shape[1]) if source_is_quantized else int(self._ffn)
        input_features = total_groups * group_size
        requested_tile_groups = int(tile_groups if tile_groups is not None else self.ffn_hidden_tile_groups)
        effective_tile_groups = max(1, min(requested_tile_groups, total_groups)) if total_groups else max(1, requested_tile_groups)
        tile_count = int(math.ceil(total_groups / effective_tile_groups)) if total_groups else 0
        tail_groups = ((total_groups - 1) % effective_tile_groups) + 1 if total_groups else 0
        tile_hidden_features = effective_tile_groups * group_size
        tail_hidden_features = tail_groups * group_size
        packed_cols_per_group = None
        if source_is_quantized:
            bits = int(getattr(self.fc2, "bits", 0) or 0)
            packed_cols_per_group = (group_size * bits // 32) if bits and (group_size * bits) % 32 == 0 else None
        source_nbytes = int(self.fc1.weight.nbytes) + int(self.fc2.weight.nbytes)
        for layer in (self.fc1, self.fc2):
            scales = getattr(layer, "scales", None)
            biases = getattr(layer, "biases", None)
            if scales is not None:
                source_nbytes += int(scales.nbytes)
            if biases is not None:
                source_nbytes += int(biases.nbytes)
        batch = int(batch_size) if batch_size is not None else None
        sequence = int(sequence_length) if sequence_length is not None else None
        elem_prefix = (batch * sequence) if batch is not None and sequence is not None else None
        gate_value_tile_nbytes = (
            int(elem_prefix * 2 * tile_hidden_features * activation_dtype_nbytes)
            if elem_prefix is not None
            else None
        )
        hidden_tile_nbytes = (
            int(elem_prefix * tile_hidden_features * activation_dtype_nbytes)
            if elem_prefix is not None
            else None
        )
        full_gate_value_nbytes = (
            int(elem_prefix * 2 * int(self._ffn) * activation_dtype_nbytes)
            if elem_prefix is not None
            else None
        )
        full_hidden_nbytes = (
            int(elem_prefix * int(self._ffn) * activation_dtype_nbytes)
            if elem_prefix is not None
            else None
        )
        accumulator_nbytes = (
            int(elem_prefix * int(self._hidden) * activation_dtype_nbytes)
            if elem_prefix is not None
            else None
        )
        return {
            "source_is_quantized_fc2": source_is_quantized,
            "uses_quantized_matmul_fc1_output_row_slices": getattr(self.fc1, "scales", None) is not None,
            "uses_quantized_matmul_fc2_input_group_slices": source_is_quantized,
            "ffn_hidden_features": int(self._ffn),
            "output_hidden_features": int(self._hidden),
            "fc2_input_features_from_quant_groups": input_features,
            "fc2_input_matches_ffn_hidden": input_features == int(self._ffn),
            "group_size": group_size,
            "bits": int(getattr(self.fc2, "bits", 0) or 0),
            "mode": str(getattr(self.fc2, "mode", "none")),
            "total_input_groups": total_groups,
            "requested_tile_groups": requested_tile_groups,
            "effective_tile_groups": effective_tile_groups,
            "tile_count": tile_count,
            "tail_tile_groups": tail_groups,
            "tile_hidden_features": tile_hidden_features,
            "tail_hidden_features": tail_hidden_features,
            "packed_cols_per_group": packed_cols_per_group,
            "packed_cols_per_full_tile": (effective_tile_groups * packed_cols_per_group) if packed_cols_per_group is not None else None,
            "materializes_full_fc1_gate_value": False,
            "materializes_full_swiglu_hidden": False,
            "learned_fc2_bias_added_once_after_accumulation": "bias" in self.fc2,
            "partial_accumulation_dtype": "float32 accumulator, cast back to partial projection dtype at the end",
            "persistent_full_dense_allocation": False,
            "packed_quantized_source_nbytes_fc1_plus_fc2": source_nbytes,
            "batch_size": batch,
            "sequence_length": sequence,
            "activation_dtype_nbytes_assumption": int(activation_dtype_nbytes),
            "max_tile_gate_value_nbytes": gate_value_tile_nbytes,
            "max_tile_hidden_nbytes": hidden_tile_nbytes,
            "full_gate_value_nbytes_baseline": full_gate_value_nbytes,
            "full_hidden_nbytes_baseline": full_hidden_nbytes,
            "accumulator_nbytes": accumulator_nbytes,
        }

    def _hidden_tile_stream_forward(self, x: mx.array) -> mx.array:
        """Stream FFN hidden channels through FC1/SwiGLU/FC2 input-group tiles.

        This candidate intentionally changes only the FFN hidden-axis schedule.  It preserves FC1
        row order and FC2 input-group order, but it can still fail strict parity because FC2's K
        reduction is split into partial reductions that are accumulated outside MLX's single QMM.
        """
        fc2_scales = getattr(self.fc2, "scales", None)
        if fc2_scales is not None:
            group_size = int(getattr(self.fc2, "group_size"))
            total_groups = int(fc2_scales.shape[1])
            if total_groups * group_size != int(self._ffn):
                raise ValueError(
                    "fc2 quantized input groups do not match FFN hidden size: "
                    f"{total_groups} * {group_size} != {int(self._ffn)}"
                )
        else:
            group_size = 1
            total_groups = int(self._ffn)

        tile_groups = int(self.ffn_hidden_tile_groups or 0)
        if tile_groups <= 0:
            raise ValueError(f"ffn_hidden_tile_groups must be positive, got {tile_groups}")
        tile_groups = min(tile_groups, total_groups)

        fc1_input = self._pre_fc1_input(x, lora=None)
        # Keep the upstream norm/AdaLN graph from being replayed for every hidden tile.
        mx.eval(fc1_input)
        mx.synchronize()

        accumulator: mx.array | None = None
        partial_dtype = fc1_input.dtype
        for group_start in range(0, total_groups, tile_groups):
            group_stop = min(group_start + tile_groups, total_groups)
            start = group_start * group_size
            stop = group_stop * group_size
            gate = linear_output_row_slice_projection(self.fc1, fc1_input, start, stop)
            value = linear_output_row_slice_projection(self.fc1, fc1_input, int(self._ffn) + start, int(self._ffn) + stop)
            hidden = nn.silu(gate) * value
            partial = linear_input_range_partial_projection(self.fc2, hidden, start, stop)
            partial_dtype = partial.dtype
            partial_f32 = partial.astype(mx.float32)
            accumulator = partial_f32 if accumulator is None else accumulator + partial_f32
            # Force each tile's gate/value/hidden/partial to retire before the next tile is built.
            mx.eval(accumulator)
            mx.synchronize()
            del gate, value, hidden, partial, partial_f32

        if accumulator is None:
            raise ValueError("cannot run hidden-tile FFN stream with zero input groups")
        projected = accumulator.astype(partial_dtype)
        if "bias" in self.fc2:
            projected = projected + self.fc2.bias
        return projected

    def _fc2_project(self, hidden: mx.array, lora=None) -> mx.array:
        hidden = self._pre_fc2_hidden(hidden, lora=lora)
        if self.use_ffn_fc2_input_chunked_qmm_candidate and lora is None:
            return quantized_matmul_input_chunked_projection(self.fc2, hidden, self.ffn_fc2_input_chunk_groups)
        if self.use_ffn_fc2_tiled_dense_dequant_candidate and lora is None:
            return tiled_dense_linear_projection(self.fc2, hidden, self.ffn_fc2_tiled_output_channels)
        if self.use_ffn_fc2_dense_dequant_candidate and lora is None:
            return dense_linear_projection(self.fc2, hidden, self._fc2_dense_dequant_weight())
        if (self.use_ffn_2d_projection_candidate or self.use_ffn_fc2_rank2_qmm_candidate) and lora is None:
            return linear_rank3_input_as_rank2(self.fc2, hidden)
        return project_linear(self.fc2, hidden)

    def _swiglu_hidden(self, fused: mx.array, x: mx.array | None = None, lora=None) -> mx.array:
        if self.use_ffn_metal_swiglu_candidate and lora is None and not self.use_mx_split_swiglu_candidate:
            return swiglu_from_fused_metal(fused, self._ffn)
        if self.use_mx_split_swiglu_candidate:
            gate, value = mx.split(fused, 2, axis=-1)
        else:
            gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
        if lora is not None:
            if x is None:
                raise ValueError("LoRA SwiGLU path requires the pre-projection input")
            if lora.has("mlp.fc1"):
                # Native H3 uses the same fused [gate; value] order as the base projection.
                gate_delta, value_delta = mx.split(lora.apply("mlp.fc1", x), 2, axis=-1)
            else:
                # Diffusers stores the fused LoRA output as [value; gate].
                value_delta, gate_delta = mx.split(lora.apply("ff.net.0.proj", x), 2, axis=-1)
            gate = gate + gate_delta
            value = value + value_delta
        return nn.silu(gate) * value

    def _ffn_baseline_subgraph(self, x: mx.array) -> mx.array:
        """Run the reference ``fc1 -> SwiGLU -> fc2`` branch with no opt-in variants.

        The FFN subgraph compile candidate wraps exactly this method so it changes only MLX graph
        scheduling around the two projections and native SwiGLU.  It intentionally bypasses other
        FFN experiment flags; the harness enables it as a single variable and the LoRA path falls
        back to the normal uncompiled implementation.
        """
        fused = self.fc1(x)
        gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
        return self.fc2(nn.silu(gate) * value)

    def _ffn_subgraph_compile_source_key(self):
        def layer_key(layer: nn.Module):
            scales = getattr(layer, "scales", None)
            biases = getattr(layer, "biases", None)
            return (
                id(layer.weight),
                id(scales),
                id(biases),
                tuple(layer.weight.shape),
                tuple(scales.shape) if scales is not None else None,
                tuple(biases.shape) if biases is not None else None,
                str(layer.weight.dtype),
                str(scales.dtype) if scales is not None else None,
                str(biases.dtype) if biases is not None else None,
                int(getattr(layer, "group_size", 0) or 0),
                int(getattr(layer, "bits", 0) or 0),
                str(getattr(layer, "mode", "none")),
            )

        return (int(self._hidden), int(self._ffn), layer_key(self.fc1), layer_key(self.fc2))

    def clear_ffn_subgraph_compile_cache(self) -> None:
        """Drop the cached ``mx.compile`` wrapper used only by the FFN subgraph probe."""
        self._ffn_subgraph_compile_cache_key = None
        self._ffn_subgraph_compile_cache = None

    def _ffn_subgraph_compiled(self):
        compile_fn = getattr(mx, "compile", None)
        if compile_fn is None:
            raise RuntimeError("mx.compile is unavailable; cannot run FFN subgraph compile candidate")
        key = self._ffn_subgraph_compile_source_key()
        compiled = self._ffn_subgraph_compile_cache if self._ffn_subgraph_compile_cache_key == key else None
        if compiled is None:
            def forward(z: mx.array) -> mx.array:
                return self._ffn_baseline_subgraph(z)

            compiled = compile_fn(forward)
            self._ffn_subgraph_compile_cache_key = key
            self._ffn_subgraph_compile_cache = compiled
        return compiled

    def ffn_subgraph_compile_cache_info(self) -> dict[str, object]:
        compile_fn = getattr(mx, "compile", None)
        return {
            "mx_compile_available": compile_fn is not None,
            "cached": self._ffn_subgraph_compile_cache is not None,
            "cache_key_set": self._ffn_subgraph_compile_cache_key is not None,
            "scope": "FeedForward-only fc1 -> native SwiGLU -> fc2 branch",
            "lora_fallback": True,
            "combines_other_ffn_candidates": False,
        }

    def _sequence_chunked_forward(self, x: mx.array) -> mx.array:
        """Run baseline ``fc1 -> SwiGLU -> fc2`` over contiguous sequence chunks.

        The sequence axis is row-independent for both FFN projections and SwiGLU, so this opt-in
        path preserves row order and math while reducing the largest temporary ``fc1``/hidden
        tensors from ``[B,S,*]`` to ``[B,C,*]`` chunks.  It intentionally uses the baseline slice
        SwiGLU and direct rank-3 projections, rather than combining with other FFN candidates.
        """
        if len(x.shape) != 3:
            fused = self.fc1(x)
            gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
            return self.fc2(nn.silu(gate) * value)

        batch, sequence, _hidden = x.shape
        chunk_size = int(self.ffn_sequence_chunk_size or 0)
        if chunk_size <= 0 or int(sequence) <= chunk_size:
            fused = self.fc1(x)
            gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
            return self.fc2(nn.silu(gate) * value)

        chunks = []
        for start in range(0, int(sequence), chunk_size):
            stop = min(start + chunk_size, int(sequence))
            x_chunk = x[:, start:stop, :]
            fused = self.fc1(x_chunk)
            gate, value = fused[..., : self._ffn], fused[..., self._ffn :]
            chunks.append(self.fc2(nn.silu(gate) * value))
        out = mx.concatenate(chunks, axis=1)
        return out.reshape(batch, sequence, out.shape[-1])

    def __call__(self, x: mx.array, lora=None) -> mx.array:
        if self.use_ffn_subgraph_compile_candidate and lora is None:
            return profiled_call("ffn.subgraph_compiled", "ffn_swiglu", lambda: self._ffn_subgraph_compiled()(x))
        if self.use_ffn_sequence_chunk_candidate and lora is None:
            return profiled_call("ffn.sequence_chunked", "ffn_swiglu", lambda: self._sequence_chunked_forward(x))
        if self.use_ffn_hidden_tile_stream_candidate and lora is None:
            return profiled_call("ffn.hidden_tile_stream", "ffn_swiglu", lambda: self._hidden_tile_stream_forward(x))
        if self.use_ffn_fc1_split_gate_value_quantized_qmm_candidate and lora is None:
            fc1_input = self._pre_fc1_input(x, lora=lora)
            gate, value = profiled_call(
                "ffn.fc1_split_gate_value_projection",
                "linear_projection",
                lambda: self._fc1_gate_value_from_prepared_input(fc1_input),
                metadata={"sequence_length": x.shape[-2], "hidden_size": self._hidden, "ffn_hidden_size": self._ffn},
            )
            hidden = profiled_call(
                "ffn.swiglu",
                "ffn_swiglu",
                lambda: nn.silu(gate) * value,
                metadata={"sequence_length": x.shape[-2], "ffn_hidden_size": self._ffn},
            )
        else:
            fused = profiled_call(
                "ffn.fc1_projection",
                "linear_projection",
                lambda: self._fc1_project(x, lora=lora),
                metadata={"sequence_length": x.shape[-2], "hidden_size": self._hidden, "ffn_hidden_size": self._ffn},
            )
            hidden = profiled_call(
                "ffn.swiglu",
                "ffn_swiglu",
                lambda: self._swiglu_hidden(fused, x=x, lora=lora),
                metadata={"sequence_length": x.shape[-2], "ffn_hidden_size": self._ffn},
            )
        projected = profiled_call(
            "ffn.fc2_projection",
            "linear_projection",
            lambda: self._fc2_project(hidden, lora=lora),
            metadata={"sequence_length": x.shape[-2], "hidden_size": self._hidden, "ffn_hidden_size": self._ffn},
        )
        if lora is None:
            return projected
        target = "mlp.fc2" if lora.has("mlp.fc2") else "ff.net.2"
        return projected + lora.apply(target, hidden)


class AdaLayerNormModulation(nn.Module):
    """Projects the shared timestep embedding into one block's six modulation parameters.

    ``(num_timesteps, time_embed_dim)`` -> six tensors of shape
    ``(num_timesteps * MODALITY_NUM, hidden_size)`` in
    ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`` order. Row layout is
    ``[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]``, which
    ``timestep_indices * MODALITY_NUM + token_tags`` addresses.

    One projection is shared by ``norm1`` and ``norm2`` and by the three modalities, so it cannot
    be folded into either norm.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.linear = nn.Linear(config.time_embed_dim, config.adaln_out_features, bias=True)

    def __call__(self, temb: mx.array, lora=None) -> tuple[mx.array, ...]:
        # Activate at `temb`'s own (float32) precision, cast to the projection's dtype after.
        activated = nn.silu(temb).astype(param_dtype(self.linear))
        h = project_linear(self.linear, activated)
        if lora is not None and lora.has("adaln_proj.linear"):
            h = h + lora.apply("adaln_proj.linear", activated)
        h = h.reshape(-1, 6 * self.hidden_size)
        return tuple(h[..., i * self.hidden_size : (i + 1) * self.hidden_size] for i in range(6))


class AdaLayerNormModulationOut(nn.Module):
    """``final_layer.adaln_proj`` — the ``linear`` sub-name matches the checkpoint."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.linear = nn.Linear(config.time_embed_dim, config.final_adaln_out_features, bias=True)


class FinalLayer(nn.Module):
    """Checkpoint's ``final_layer``: shared modulated norm plus the two output heads.

    The modulation table holds one row per *timestep* and is addressed per row of the packed
    sequence. The two halves of the projection are ``shift`` then ``scale``.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = AdaLayerNormModulationOut(config)
        self.video_out = nn.Linear(config.hidden_size, config.video_patch_dim, bias=True)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_latents_dim, bias=True)
        self.hidden_size = config.hidden_size

    def norm_out(
        self,
        x: mx.array,
        temb: mx.array,
        timestep_indices: mx.array,
        lora=None,
    ) -> mx.array:
        activated = nn.silu(temb).astype(param_dtype(self.adaln_proj.linear))
        h = self.adaln_proj.linear(activated)
        if lora is not None and lora.has("final_layer.adaln_proj.linear"):
            h = h + lora.apply("final_layer.adaln_proj.linear", activated)
        shift, scale = h[..., : self.hidden_size], h[..., self.hidden_size :]
        x = self.norm(x)
        return x * (1.0 + scale[timestep_indices]) + shift[timestep_indices]


class TokenRefinerBlock(nn.Module):
    """Plain pre-norm block used to refine the projected text stream. No AdaLN, no rotary."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def __call__(self, x: mx.array, lora=None) -> mx.array:
        x = x + self.attn(self.norm1(x), lora=lora)
        return x + self.mlp(self.norm2(x), lora=lora)


class TokenRefiner(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.blocks = [TokenRefinerBlock(config) for _ in range(config.token_refiner_num_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def __call__(self, x: mx.array, loras=None) -> mx.array:
        for index, block in enumerate(self.blocks):
            lora = loras[index] if loras is not None else None
            x = block(x, lora)
        return self.final_norm(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward, each modulated by AdaLN parameters selected per row."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = FeedForward(config)
        self.adaln_proj = AdaLayerNormModulation(config)
        # Disabled-by-default modulation/residual scheduling probes.  Default behavior remains the
        # reference table gathers and pointwise residual arithmetic at the point of use.
        self.use_packed_adaln_gather_candidate = False
        self.use_indexed_adaln_affine_metal_candidate = False
        self.use_indexed_gated_residual_metal_candidate = False

    def __call__(
        self,
        x: mx.array,
        modulation: tuple[mx.array, ...],
        adaln_indices: mx.array,
        rotary: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
        lora=None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        if self.use_packed_adaln_gather_candidate:
            (
                shift_msa_rows,
                scale_msa_rows,
                gate_msa_rows,
                shift_mlp_rows,
                scale_mlp_rows,
                gate_mlp_rows,
            ) = gather_packed_modulation_rows(modulation, adaln_indices)
            h = self.norm1(x)
            h = h * (1.0 + scale_msa_rows) + shift_msa_rows
            x = x + gate_msa_rows * self.attn(h, rotary, mask, lora)

            h = self.norm2(x)
            h = h * (1.0 + scale_mlp_rows) + shift_mlp_rows
            return x + gate_mlp_rows * self.mlp(h, lora)

        def norm1_adaln() -> mx.array:
            h = self.norm1(x)
            if self.use_indexed_adaln_affine_metal_candidate and lora is None:
                return indexed_adaln_affine_metal(h, scale_msa, shift_msa, adaln_indices)
            return h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]

        h = profiled_call(
            "block.norm1_adaln_msa",
            "adaln_norm_residual",
            norm1_adaln,
            metadata={"sequence_length": x.shape[-2], "hidden_size": x.shape[-1]},
        )
        attn_out = self.attn(h, rotary, mask, lora)

        def msa_residual() -> mx.array:
            if self.use_indexed_gated_residual_metal_candidate and lora is None:
                return indexed_gated_residual_metal(x, gate_msa, adaln_indices, attn_out)
            return x + gate_msa[adaln_indices] * attn_out

        x = profiled_call(
            "block.msa_gated_residual",
            "adaln_norm_residual",
            msa_residual,
            metadata={"sequence_length": x.shape[-2], "hidden_size": x.shape[-1]},
        )

        def norm2_adaln() -> mx.array:
            h = self.norm2(x)
            if self.use_indexed_adaln_affine_metal_candidate and lora is None:
                return indexed_adaln_affine_metal(h, scale_mlp, shift_mlp, adaln_indices)
            return h * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]

        h = profiled_call(
            "block.norm2_adaln_mlp",
            "adaln_norm_residual",
            norm2_adaln,
            metadata={"sequence_length": x.shape[-2], "hidden_size": x.shape[-1]},
        )
        mlp_out = self.mlp(h, lora)

        def mlp_residual() -> mx.array:
            if self.use_indexed_gated_residual_metal_candidate and lora is None:
                return indexed_gated_residual_metal(x, gate_mlp, adaln_indices, mlp_out)
            return x + gate_mlp[adaln_indices] * mlp_out

        return profiled_call(
            "block.mlp_gated_residual",
            "adaln_norm_residual",
            mlp_residual,
            metadata={"sequence_length": x.shape[-2], "hidden_size": x.shape[-1]},
        )


def apply_dense_dequant_profile_to_block(
    block: TransformerBlock,
    profile: str | None,
    *,
    attention_qkv_tile_size: int = 2048,
    ffn_fc2_tile_size: int = 1024,
    attention_out_tile_size: int = 2048,
) -> str:
    """Apply the disabled-by-default dense-dequant block profile.

    The profiles are opt-in/provenance-only after generation A/Bs rejected the
    broader dense-dequant combinations for default deployment.  The narrow
    profiles enable exactly one transient tiled dense-dequant route, either
    ``attn.qkv_proj`` or ``mlp.fc2`` (tile size defaults to 1024 for fc2); the
    combined profiles add tiled dense-dequant for ``mlp.fc2`` plus exactly one
    ``attn.out_proj`` route.  They are designed for the low-memory streamed-
    block path, so resident ``out_proj`` dequantization is at most one block
    scoped.
    """
    selected = normalize_dense_dequant_profile(profile)
    for name, value in (
        ("attention_qkv_tile_size", attention_qkv_tile_size),
        ("ffn_fc2_tile_size", ffn_fc2_tile_size),
        ("attention_out_tile_size", attention_out_tile_size),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    block_candidate_flags = (
        "use_packed_adaln_gather_candidate",
        "use_indexed_adaln_affine_metal_candidate",
        "use_indexed_gated_residual_metal_candidate",
    )
    attention_candidate_flags = (
        "use_pre_qkv_contiguous_candidate",
        "use_qkv_2d_projection_candidate",
        "use_out_2d_projection_candidate",
        "use_out_dense_dequant_candidate",
        "use_out_tiled_dense_dequant_candidate",
        "use_qkv_tiled_dense_dequant_candidate",
        "use_qkv_headgroup_row_sliced_qmm_candidate",
        "use_qkv_input_chunked_qmm_candidate",
        "use_qkv_pretranspose_layout_candidate",
        "use_qkv_rmsnorm_sdpa_metal_candidate",
        "use_qkv_rmsnorm_rotary_sdpa_metal_candidate",
        "use_rotary_qk_metal_candidate",
        "use_pre_sdpa_contiguous_candidate",
        "use_sdpa_head_batch_rank3_candidate",
        "use_sdpa_headgroup_split_candidate",
        "use_sdpa_out_layout_metal_candidate",
        "use_pre_out_proj_contiguous_candidate",
    )
    ffn_candidate_flags = (
        "use_mx_split_swiglu_candidate",
        "use_ffn_2d_projection_candidate",
        "use_ffn_fc1_rank2_qmm_candidate",
        "use_ffn_fc1_split_gate_value_quantized_qmm_candidate",
        "use_ffn_fc2_rank2_qmm_candidate",
        "use_ffn_fc1_dense_dequant_candidate",
        "use_ffn_fc1_tiled_dense_dequant_candidate",
        "use_ffn_fc2_dense_dequant_candidate",
        "use_ffn_fc2_tiled_dense_dequant_candidate",
        "use_ffn_fc2_input_chunked_qmm_candidate",
        "use_ffn_hidden_tile_stream_candidate",
        "use_ffn_metal_swiglu_candidate",
        "use_ffn_sequence_chunk_candidate",
        "use_ffn_pre_fc1_contiguous_candidate",
        "use_ffn_pre_fc2_contiguous_candidate",
        "use_ffn_subgraph_compile_candidate",
    )

    for flag in block_candidate_flags:
        setattr(block, flag, False)
    for flag in attention_candidate_flags:
        setattr(block.attn, flag, False)
    for flag in ffn_candidate_flags:
        setattr(block.mlp, flag, False)

    block.attn.qkv_tiled_output_channels = int(attention_qkv_tile_size)
    block.mlp.ffn_fc2_tiled_output_channels = int(ffn_fc2_tile_size)
    block.attn.out_tiled_output_channels = int(attention_out_tile_size)

    qkv_tiled = selected in (
        DENSE_DEQUANT_PROFILE_QKV_ONLY_TILED,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
    )
    fc2_tiled = selected in (
        DENSE_DEQUANT_PROFILE_FFN_FC2_TILED,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT,
        DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED,
    )
    block.attn.use_qkv_tiled_dense_dequant_candidate = qkv_tiled
    block.mlp.use_ffn_fc2_tiled_dense_dequant_candidate = fc2_tiled
    block.attn.use_out_dense_dequant_candidate = selected == DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT
    block.attn.use_out_tiled_dense_dequant_candidate = selected == DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_TILED

    if selected != DENSE_DEQUANT_PROFILE_QKV_FC2_OUT_RESIDENT:
        clear_out_cache = getattr(block.attn, "clear_out_dense_dequant_cache", None)
        if clear_out_cache is not None:
            clear_out_cache()
    clear_fc2_cache = getattr(block.mlp, "clear_fc2_dense_dequant_cache", None)
    if clear_fc2_cache is not None:
        clear_fc2_cache()
    return selected


class MiniMaxH3DiT(nn.Module):
    """The MiniMax-H3 joint video + audio diffusion transformer.

    The caller builds the packed layout: patchified video latents, row order, the ``(t, h, w)``
    position grid, per-row modality tags and per-row timestep indices. See ``packing.py``.
    """

    def __init__(self, config: DiTConfig, *, build_blocks: bool = True):
        super().__init__()
        self.config = config

        # 1. Per-modality input projections.
        self.video_patch_proj = nn.Linear(config.video_patch_dim, config.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, bias=True)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, bias=True)

        # 2. Timestep embedding, shared by every AdaLN projection.
        self.time_embedder = TimestepEmbedder(config)

        # 3. Text stream refiner.
        self.token_refiner = TokenRefiner(config)

        # 4. The block stack.
        self.blocks = (
            [TransformerBlock(config) for _ in range(config.num_layers)]
            if build_blocks
            else []
        )

        # 5. Shared output norm and the two per-modality heads. Both heads run over every row;
        #    the rows of each modality are selected afterwards.
        self.final_layer = FinalLayer(config)

        # Rotary is a computed buffer, not a parameter (`rope.inv_freq` is recomputed bit-exactly).
        self.rope = RotaryPosEmbed3D(config)

    def precompute_text_conditioning(
        self,
        text_embeds: mx.array,
        block_provider: "QuantizedBlockProvider | None" = None,
    ) -> mx.array:
        """Project and refine prompt rows once for opt-in reuse across denoising steps.

        This is the exact text-conditioning work normally performed at the start of every
        :meth:`__call__`.  It intentionally caches only the refined text tensor, not packed-row
        layout, rotary state, timestep-dependent AdaLN data, or media latents.
        """
        text = self.condition_proj(text_embeds.astype(param_dtype(self.condition_proj)))
        refiner_loras = block_provider.refiner_loras if block_provider is not None else None
        return self.token_refiner(text, refiner_loras)

    def __call__(
        self,
        video_latents: mx.array,
        audio_latents: mx.array,
        text_embeds: mx.array | None,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
        modulation_cache: "ModulationCache | None" = None,
        mask: mx.array | None = None,
        block_cache: BlockResidualCache | None = None,
        block_cache_sigma: float = 0.0,
        block_cache_step: int = 0,
        block_cache_total_steps: int = 1,
        block_provider: "QuantizedBlockProvider | None" = None,
        refined_text: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Predict the video and audio velocity for one packed sequence.

        Args:
            video_latents: ``(B, num_video_tokens, video_patch_dim)`` patchified rows, ordered to
                match ``video_indices`` (conditioning rows included).
            audio_latents: ``(B, num_audio_tokens, audio_latents_dim)`` ordered as ``audio_indices``.
            text_embeds: ``(B, num_text_tokens, text_dim)`` ordered as ``text_indices``; may be
                ``None`` only when ``refined_text`` supplies the already-refined prompt rows.
            timestep: ``(num_timesteps,)`` the *distinct* noise levels present, unscaled in [0, 1].
            timestep_indices: ``(seq_len,)`` index into ``timestep`` for every row.
            token_tags: ``(seq_len,)`` modality per row (0 video, 1 text, 2 audio, -1 padding).
            position_ids: ``(seq_len, 3)`` the ``(t, h, w)`` rotary coordinates per row.
            modulation_cache: optional precomputed AdaLN table; when supplied the ``adaln_proj``
                weights are never read, which is what lets them be dropped at inference time.
            refined_text: optional output of :meth:`precompute_text_conditioning`. When supplied,
                ``condition_proj`` and ``token_refiner`` are skipped for this DiT call.

        Returns:
            ``(video_velocity, audio_velocity)`` in the row order of ``video_indices`` /
            ``audio_indices``.
        """
        seq_len = position_ids.shape[0]
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be (seq_len, 3), got {position_ids.shape}.")
        if token_tags.shape != (seq_len,) or timestep_indices.shape != (seq_len,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must be (seq_len,) matching `position_ids`, got "
                f"{token_tags.shape} and {timestep_indices.shape} for seq_len={seq_len}."
            )

        rotary = profiled_call(
            "dit.rotary_embedding",
            "attention_sdpa",
            lambda: self.rope(position_ids),
            metadata={"sequence_length": seq_len},
        )

        # 1. Project each modality and scatter the rows into the packed buffer. The text stream
        #    sets the dtype of the packed sequence.
        video_embeds = profiled_call(
            "dit.input_video_projection",
            "linear_projection",
            lambda: self.video_patch_proj(video_latents.astype(param_dtype(self.video_patch_proj))),
            metadata={"rows": video_latents.shape[-2], "input_features": video_latents.shape[-1]},
        )
        audio_embeds = profiled_call(
            "dit.input_audio_projection",
            "linear_projection",
            lambda: self.audio_patch_proj(audio_latents.astype(param_dtype(self.audio_patch_proj))),
            metadata={"rows": audio_latents.shape[-2], "input_features": audio_latents.shape[-1]},
        )
        if refined_text is None:
            if text_embeds is None:
                raise ValueError("`text_embeds` is required when `refined_text` is not supplied.")
            text = profiled_call(
                "dit.text_conditioning_projection_refiner",
                "text_conditioning",
                lambda: self.precompute_text_conditioning(text_embeds, block_provider=block_provider),
                metadata={"text_tokens": text_embeds.shape[-2], "text_dim": text_embeds.shape[-1]},
            )
        else:
            if len(refined_text.shape) != 3:
                raise ValueError(f"`refined_text` must be rank-3, got {refined_text.shape}.")
            if refined_text.shape[1] != text_indices.shape[0]:
                raise ValueError(
                    "`refined_text` token count must match `text_indices`, got "
                    f"{refined_text.shape[1]} vs {text_indices.shape[0]}."
                )
            text = refined_text

        B = text.shape[0]

        def pack_rows() -> mx.array:
            packed = mx.zeros((B, seq_len, text.shape[-1]), dtype=text.dtype)
            packed[:, text_indices] = text
            packed[:, video_indices] = video_embeds.astype(text.dtype)
            packed[:, audio_indices] = audio_embeds.astype(text.dtype)
            return packed

        x = profiled_call(
            "dit.pack_modalities",
            "other_forward",
            pack_rows,
            metadata={"sequence_length": seq_len},
        )

        # 2. One timestep embedding per distinct noise level, shared by all AdaLN projections.
        temb = profiled_call(
            "dit.time_embedder",
            "adaln_norm_residual",
            lambda: self.time_embedder(timestep_embedding(timestep, self.config.timestep_input_dim)),
            metadata={"distinct_timestep_count": timestep.shape[0]},
        )

        # 3. Row -> AdaLN table row. `maximum(tags, 0)` mirrors the reference clamp: padding rows
        #    carry tag -1 and must not index backwards. They never reach the outputs.
        adaln_indices = timestep_indices * MODALITY_NUM + mx.maximum(token_tags, 0)

        def run_range(hidden: mx.array, start: int, end: int) -> mx.array:
            for i in range(start, end):
                block = (
                    profiled_call(
                        "streaming.block_load",
                        "load_overhead",
                        lambda i=i: block_provider.load_block(
                            i,
                            include_adaln=modulation_cache is None,
                        ),
                        eval_output=False,
                        metadata={"block_index": i},
                    )
                    if block_provider is not None
                    else self.blocks[i]
                )
                lora = block_provider.current_lora if block_provider is not None else None
                modulation = (
                    modulation_cache.get(i)
                    if modulation_cache is not None
                    else profiled_call(
                        "block.adaln_projection",
                        "adaln_norm_residual",
                        lambda: block.adaln_proj(temb, lora=lora),
                        metadata={"block_index": i, "distinct_timestep_count": timestep.shape[0]},
                    )
                )
                hidden = block(hidden, modulation, adaln_indices, rotary, mask, lora)
                if block_provider is not None:
                    # Materialize before the reusable slot is rebound to the next block.
                    mx.eval(hidden)
            return hidden

        block_count = block_provider.block_count if block_provider is not None else len(self.blocks)
        if block_cache is None:
            x = run_range(x, 0, block_count)
        else:
            x = block_cache.run(
                x,
                block_count=block_count,
                run_range=run_range,
                sigma=block_cache_sigma,
                step_index=block_cache_step,
                total_steps=block_cache_total_steps,
            )

        # 4. Both heads run over every row, then each modality's rows are selected.
        x = profiled_call(
            "dit.final_norm_adaln",
            "adaln_norm_residual",
            lambda: self.final_layer.norm_out(
                x,
                temb,
                timestep_indices,
                lora=(getattr(block_provider, "final_lora", None) if block_provider is not None else None),
            ),
            metadata={"sequence_length": seq_len, "hidden_size": self.config.hidden_size},
        )
        video_out = profiled_call(
            "dit.final_video_head",
            "final_heads",
            lambda: self.final_layer.video_out(x.astype(param_dtype(self.final_layer.video_out))),
            metadata={"sequence_length": seq_len},
        )
        audio_out = profiled_call(
            "dit.final_audio_head",
            "final_heads",
            lambda: self.final_layer.audio_out(x.astype(param_dtype(self.final_layer.audio_out))),
            metadata={"sequence_length": seq_len},
        )
        return video_out[:, video_indices], audio_out[:, audio_indices]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .streaming import QuantizedBlockProvider
