"""Estimate how much memory a downloaded model needs: unified memory on MLX, GPU and system memory on CUDA.

Estimates read only safetensors headers (tensor dtypes and shapes), never the weights, and mirror how
each backend holds each component in memory. They are deliberately rough: a fixed working-memory
margin stands in for activations in image models, and the LTX MLX clip estimate uses working-memory
curves measured over the clip's size and length. LoRAs are not counted.
"""

from __future__ import annotations

import functools
import json
import math
import os
import struct
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.model_files import (
    component_of,
    find_local_model_dir,
    is_ideogram4_model,
    is_krea2_model,
    is_ltx_mlx_layout,
    ltx_mlx_text_encoder_repo,
    ltx_mlx_transformer_file,
    ltx_mlx_upsampler_file,
    model_weight_files,
)
from zvisiongenerator.utils.stored_quant import stored_quant_bits

GIB = 1024**3
_IMAGE_WORKING_BYTES = int(1.5 * GIB)
# Ideogram 4 Instant on MLX, above its weights at 1024x1024: the VAE decode peaks 15.6-16.2 GiB over the weights at
# every level (FP8, q8, q4), measured on a 32 GB M5; its denoising alone needs 4-5 GiB.
_IDEOGRAM4_MLX_WORKING_BYTES = 16 * GIB
# Apple's recommended working set is conservative: a modest overshoot is absorbed by memory compression
# without a noticeable slowdown.
_FITS_RATIO = 1.1
# MLX's own default memory limit is 1.5x Apple's recommended working set: up to there it keeps allocating and
# lets macOS compress and swap, so models in that range run, at the cost of slowing the rest of the Mac.
_TIGHT_LIMIT_RATIO = 1.5
_MAX_HEADER_BYTES = 100 * 1024 * 1024
# CUDA (diffusers backend). Unquantized and q8 weights stay in system memory and stream to the GPU block by
# block; the GPU holds the blocks in flight plus activations (up to about 4 GB at 1024x1024).
_CUDA_STREAMED_GPU_BYTES = 4 * GIB
_CUDA_RESIDENT_WORKING_BYTES = int(1.5 * GIB)
_CUDA_SYSTEM_WORKING_BYTES = int(1.5 * GIB)
# bitsandbytes NF4 packs two weights per byte plus a double-quantized scale per 64-weight block (~4.13 bits).
_NF4_BYTES_PER_WEIGHT = 4.13 / 8
_CUDA_QUANTIZED_COMPONENTS = ("text_encoder", "transformer")
# A CUDA model fits while its GPU need stays under 90% of VRAM and its weights under 80% of system memory,
# leaving the rest to the desktop and other apps. Weights beyond that stream from disk, which is much slower.
_CUDA_GPU_FITS_RATIO = 0.9
_CUDA_SYSTEM_FITS_RATIO = 0.8
# LTX on CUDA: beyond the weights, the process holds the CUDA runtime, the VAEs' and tokenizer's CPU side and the
# copies a load stages. Measured at q4 on a 10 GB GPU: peaks of 20-21 GB against 14.3 GiB of weights.
_LTX_CUDA_RUNTIME_BYTES = 6 * GIB
# LTX on CUDA, per clip. With the transformer streamed (low-memory mode), the peak allocated GPU memory is flat up to a
# knee, then grows linearly with the token count (the float32 temporaries outside the chunked layers). Measured on a
# 10 GB card with 2-step runs: q4 4.40 GiB from 5k to 20k tokens, then 6.44 at 45.6k and 7.21 at 53.0k; q8 3.61 GiB at
# 5k and 6.30 at 45.6k. The same slope fits both levels and predicts 7.6 GiB at 56.7k tokens, where XL 15 s ran out of memory.
_LTX_CUDA_GPU_PLATEAU_BYTES = {4: int(4.40 * GIB), 8: int(3.61 * GIB)}
_LTX_CUDA_GPU_KNEE_TOKENS = {4: 26000, 8: 19700}
_LTX_CUDA_GPU_BYTES_PER_TOKEN = int(1.04e-4 * GIB)
# The two-stage upscale also holds the latent upsampler on the GPU: a small clip peaked at 5.04 GiB at q4, 0.64 over its plateau.
_LTX_CUDA_UPSCALE_GPU_BYTES = int(0.64 * GIB)
# Beyond the allocated peak, the caching allocator fragments (reserved was 1.0-2.2 GiB above allocated) and the CUDA
# context and desktop take their share; with it XL 14 s (7.21 GiB) is tight and XL 15 s (7.6 GiB) too large on 10 GiB.
_LTX_CUDA_GPU_HEADROOM_BYTES = int(2.5 * GIB)
# The two-stage upscale also holds the upsampler and the first stage's latents: measured +3 to +5 GB of system memory.
_LTX_CUDA_UPSCALE_SYSTEM_BYTES = 4 * GIB

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}
_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
# MLX affine quantization packs weights to `bits` and stores a bf16 scale and bias per 64-weight group.
_QUANT_GROUP_OVERHEAD_BYTES = 4 / 64
# Diffusion pipelines read text-encoder hidden states, never logits, so mflux does not load the LM head.
_TEXT_ENCODER_SKIPPED_PREFIXES = ("lm_head.",)
_LTX_MLX_CONNECTOR_FILE = "connector.safetensors"
# The decoders loaded for a decode: the VAE tiler's budget limits only its activations, not these weights.
_LTX_MLX_DECODER_FILES = ("vae_decoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors")
# LTX MLX working memory beyond the weights, in GiB, from per-stage peak memory measured with ``mx.get_peak_memory``
# (one fresh process per clip). Fit on the q4 distilled LTX 2.3 model with its Gemma 4-bit text
# encoder on a 32 GiB Apple Silicon Mac, where the transformer is not streamed; Macs with 24 GiB or less stream
# it, so the estimate is conservative there. Each term is a measured absolute curve, not scaled by model.
_LTX_PROMPT_WORKING_GIB = 0.85  # Gemma + connector while the prompt is encoded.
_LTX_DENOISE_FLOOR_GIB = 1.6  # Transformer working memory at small clips.
_LTX_DENOISE_AT_REFERENCE_GIB = 1.7  # ... at _LTX_REFERENCE_TOKENS tokens,
_LTX_DENOISE_GIB_PER_TOKEN = 0.000205  # ... and growing by this per token.
_LTX_REFERENCE_TOKENS = 4928  # 704x448 at 121 frames.
_LTX_DECODE_BASE_GIB = 12.9  # Peak of a VAE decode (decoder weights included) up to _LTX_DECODE_KNEE_TOKENS tokens,
_LTX_DECODE_KNEE_TOKENS = 2048
_LTX_DECODE_GIB_PER_TOKEN = 0.00067  # ... growing by this per token beyond it, until the tiler's budget caps it.
_LTX_UPSAMPLE_WORKING_GIB = 0.5  # Transformer + spatial upsampler + this, before the second denoise stage.
_LTX_DECODE_RELEASED_GIB = 2.83  # With Low memory off, the decode peak includes weights that stay resident anyway.
_LTX_VAE_DECODE_BUDGET_ENV = "LTX2_VAE_DECODE_BUDGET_GB"  # Mirrors ltx_core_mlx's decode_budget_bytes().
_LTX_TOKEN_FRAME_STEP = 8
_LTX_TOKEN_PIXELS = 32


@dataclass(frozen=True)
class MemoryBudget:
    """The memory a model may use on this machine.

    On Apple Silicon (unified memory) *gpu_bytes* is Apple's recommended GPU working set and *system_bytes* is
    ``None``. On a discrete GPU (CUDA) *gpu_bytes* is the VRAM and *system_bytes* the system memory that
    offloaded weights stream from. *unified_bytes* is the total unified memory on Apple Silicon, when known.
    """

    gpu_bytes: int
    system_bytes: int | None = None
    unified_bytes: int | None = None


@dataclass(frozen=True)
class CudaMemoryEstimate:
    """Estimated peak GPU memory and system memory for a model on the diffusers/CUDA backend."""

    gpu_bytes: int
    system_bytes: int


@dataclass(frozen=True)
class WeightTotals:
    """Summarise the tensors of one or more safetensors files by how the loaders hold them."""

    float_matrix_elements: int = 0  # 2-D floating-point tensors: the weights quantization packs
    float_other_elements: int = 0  # other floating-point tensors (biases, norms, convolutions)
    packed_bytes: int = 0  # other non-float tensors (packed quantized weights, ints, non-2-D FP8), resident as stored
    prequantized: bool = False  # packed U32 weights: already quantized, so requested quantization is ignored
    fp8_matrix_elements: int = 0  # 2-D FP8 tensors: one byte each as stored, and mflux quantization packs them too

    def __add__(self, other: WeightTotals) -> WeightTotals:
        return WeightTotals(
            self.float_matrix_elements + other.float_matrix_elements,
            self.float_other_elements + other.float_other_elements,
            self.packed_bytes + other.packed_bytes,
            self.prequantized or other.prequantized,
            self.fp8_matrix_elements + other.fp8_matrix_elements,
        )

    @property
    def bfloat16_bytes(self) -> int:
        """Return resident bytes with floats loaded as bfloat16 and everything else as stored."""
        return (self.float_matrix_elements + self.float_other_elements) * 2 + self.fp8_matrix_elements + self.packed_bytes


def read_safetensors_totals(path: Path, skipped_prefixes: tuple[str, ...] = ()) -> WeightTotals:
    """Summarise a safetensors file from its header, cached per file version.

    Args:
        path: The safetensors file.
        skipped_prefixes: Tensor-name prefixes the loader never reads, left out of the totals.

    Raises:
        ValueError: If the file is not a readable safetensors file, or uses a dtype the estimate does not
            know (e.g. sub-byte formats): guessing its size could mislabel a model, so it is not estimated.
    """
    stat = path.stat()
    return _read_totals_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns, skipped_prefixes)


@functools.lru_cache(maxsize=512)
def _read_totals_cached(path: str, size: int, mtime_ns: int, skipped_prefixes: tuple[str, ...]) -> WeightTotals:
    del size, mtime_ns  # Cache-key only: a rewritten file gets a fresh read.
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Not a safetensors file: {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(f"Safetensors header too large in {path}")
        header = json.loads(handle.read(header_len))
    if not isinstance(header, dict):
        raise ValueError(f"Malformed safetensors header in {path}")
    matrix = other = packed = fp8_matrix = 0
    prequantized = False
    for name, spec in header.items():
        if name == "__metadata__" or name.startswith(skipped_prefixes):
            continue
        dtype = str(spec["dtype"])
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} in {path}")
        shape = [int(dim) for dim in spec["shape"]]
        elements = math.prod(shape)
        if dtype.startswith("F8_") and len(shape) == 2:
            fp8_matrix += elements
        elif dtype not in _FLOAT_DTYPES:
            packed += elements * _DTYPE_BYTES[dtype]
            prequantized = prequantized or dtype == "U32"
        elif len(shape) == 2:
            matrix += elements
        else:
            other += elements
    return WeightTotals(matrix, other, packed, prequantized, fp8_matrix)


def estimate_image_memory(model_dir: Path, quantize_levels: tuple[int | None, ...] = (None,)) -> dict[int | None, int] | None:
    """Estimate resident memory for an mflux image model loaded in bfloat16, at each quantize level.

    Mirrors ``MfluxBackend.load_model``: floating-point weights load as bfloat16, the VAE is upcast to
    float32, FP8 and pre-quantized weights stay as stored, and quantizing packs 2-D weights. Only the
    files the loader reads count (see :func:`model_weight_files`), minus the text encoder's unused LM head,
    and headers are read once for all levels. Ideogram 4 adds its measured working memory at 1024x1024, whose VAE
    decode peaks far above the weights; other families add the fixed margin.

    Args:
        model_dir: Directory holding the model's safetensors files.
        quantize_levels: Levels to estimate: ``None`` for unquantized, or 4 / 8 bits.

    Returns:
        Estimated peak bytes per level, or ``None`` when weights are missing or unreadable.
    """
    components = _image_components(model_dir)
    if components is None:
        return None
    # mflux keeps Krea 2's text encoder in bfloat16 at every quantize level.
    quantize_text_encoder = not is_krea2_model(model_dir)
    working = _IDEOGRAM4_MLX_WORKING_BYTES if is_ideogram4_model(model_dir) else _IMAGE_WORKING_BYTES
    return {
        level: int(sum(_image_component_bytes(name, totals, _component_quantize(name, level, quantize_text_encoder)) for name, totals in components.items())) + working for level in quantize_levels
    }


def estimate_cuda_image_memory(model_dir: Path, quantize_levels: tuple[int | None, ...] = (None,)) -> dict[int | None, CudaMemoryEstimate] | None:
    """Estimate GPU and system memory for a diffusers image model on CUDA, at each quantize level.

    Mirrors ``DiffusersBackend.load_model``: unquantized (bfloat16) and q8 (FP8 weight storage) keep the
    transformer and text encoder in system memory and stream them to the GPU; q4 holds them in NF4 on the GPU,
    one at a time for Krea 2 (whose NF4 pair does not fit a 10-12 GB card together). The VAE stays on the GPU.
    Ideogram 4 Instant streams its transformer at every level and keeps its pre-quantized NF4 text encoder on the
    GPU only while it encodes (see :func:`_cuda_ideogram4_estimate`). A stored quant loads at its own level whatever
    level is asked for.

    Args:
        model_dir: Directory holding the model's safetensors files.
        quantize_levels: Levels to estimate: ``None`` for unquantized, or 4 / 8 bits.

    Returns:
        Estimated peak memory per level, or ``None`` when weights are missing or unreadable.
    """
    components = _image_components(model_dir)
    if components is None:
        return None
    stored_bits = stored_quant_bits(model_dir)
    if is_ideogram4_model(model_dir):
        return {level: _cuda_ideogram4_estimate(components, stored_bits or level) for level in quantize_levels}
    # Mirrors _Q4_CPU_OFFLOAD_FAMILIES in backends/cuda/image/backend.py.
    text_encoder_offloaded = is_krea2_model(model_dir)
    return {level: _cuda_level_estimate(components, stored_bits or level, text_encoder_offloaded) for level in quantize_levels}


def ltx_video_tokens(width: int, height: int, frames: int) -> int:
    """Return the video tokens of an LTX clip: latent frames (8 frames each, plus one) times 32-pixel cells."""
    return ((frames - 1) // _LTX_TOKEN_FRAME_STEP + 1) * (width // _LTX_TOKEN_PIXELS) * (height // _LTX_TOKEN_PIXELS)


def ltx_decode_budget_bytes(unified_bytes: int | None, env: Mapping[str, str]) -> int | None:
    """Return the peak-memory budget of an LTX VAE decode, mirroring ltx_core_mlx's ``decode_budget_bytes``.

    Args:
        unified_bytes: The machine's unified memory, or ``None`` when unknown.
        env: The environment; ``LTX2_VAE_DECODE_BUDGET_GB`` (GiB) overrides the default of half of unified memory.

    Returns:
        The budget in bytes, or ``None`` when the environment sets none and unified memory is unknown.
    """
    configured = env.get(_LTX_VAE_DECODE_BUDGET_ENV)
    if configured is not None:
        try:
            return int(float(configured) * GIB)
        except ValueError:
            pass  # An unusable value: the pipeline would fail on it, the estimate falls back to the default.
    return unified_bytes // 2 if unified_bytes else None


def estimate_ltx_mlx_clip_memory(
    model_dir: Path,
    text_encoder_dir: Path,
    *,
    width: int,
    height: int,
    frames: int,
    low_memory: bool,
    upscale: bool = False,
    unified_bytes: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Estimate peak memory for generating a clip with ltx-pipelines-mlx.

    The peak is the largest stage. Weights are read from safetensors headers (bfloat16 bytes of Gemma *G*,
    connector *C*, transformer *T*, spatial upsampler *U* and decoders *D*: video VAE, audio VAE, vocoder). The
    working memory of each stage is a measured curve (the ``_LTX_*_GIB`` constants) over the video tokens of the
    final output size, which is stage 2 when upscaling::

        prompt   = G + C + prompt working memory
        denoise  = T + denoise working memory(tokens)
        decode   = min(decode budget + D, decode peak(tokens))   # the decode peak includes D
        upsample = T + U + upsample working memory              # upscale only

    With ``low_memory`` the stages run one after another and the peak is their maximum. Without it Gemma, the
    connector, the transformer and the upsampler stay resident, so their weights add to the larger of the
    denoise working memory and the decode peak less the weights it already counts. The decode budget is the VAE
    tiler's (half of unified memory, or ``LTX2_VAE_DECODE_BUDGET_GB``); it limits activations, not weights.

    The working constants were fit on the q4 distilled model on a 32 GiB Mac. Macs with 24 GiB or less stream
    the transformer, so the estimate is conservative there. It ignores memory used by other apps.

    Args:
        model_dir: The LTX MLX model directory.
        text_encoder_dir: The Gemma text-encoder directory.
        width: Output width in pixels, as aligned for the job.
        height: Output height in pixels, as aligned for the job.
        frames: Output frame count, as aligned for the job.
        low_memory: Whether the pipeline frees Gemma before loading the transformer.
        upscale: Whether the clip is upscaled 2x. Without an upsampler file in *model_dir* the upsampler is left out.
        unified_bytes: The machine's unified memory; without it the decode peak is not capped.
        env: The environment to read the decode budget from; the process environment when omitted.

    Returns:
        Estimated peak bytes, or ``None`` when the layout is not LTX MLX or weights are unreadable.
    """
    transformer_file = ltx_mlx_transformer_file(model_dir)
    if not is_ltx_mlx_layout(model_dir) or transformer_file is None:
        return None
    upsampler_file = ltx_mlx_upsampler_file(model_dir) if upscale else None
    try:
        gemma = _files_bfloat16_bytes(model_weight_files(text_encoder_dir)) / GIB
        connector = _files_bfloat16_bytes([model_dir / _LTX_MLX_CONNECTOR_FILE]) / GIB
        transformer = _files_bfloat16_bytes([transformer_file]) / GIB
        upsampler = _files_bfloat16_bytes([upsampler_file]) / GIB if upsampler_file is not None else 0.0
        decoders = _files_bfloat16_bytes([model_dir / name for name in _LTX_MLX_DECODER_FILES]) / GIB
    except OSError, ValueError, KeyError, TypeError:
        return None
    tokens = ltx_video_tokens(width, height, frames)
    denoise_work = max(_LTX_DENOISE_FLOOR_GIB, _LTX_DENOISE_AT_REFERENCE_GIB + _LTX_DENOISE_GIB_PER_TOKEN * (tokens - _LTX_REFERENCE_TOKENS))
    decode = _LTX_DECODE_BASE_GIB + _LTX_DECODE_GIB_PER_TOKEN * max(0, tokens - _LTX_DECODE_KNEE_TOKENS)
    decode_budget = ltx_decode_budget_bytes(unified_bytes, os.environ if env is None else env)
    if decode_budget is not None:
        decode = min(decode, decode_budget / GIB + decoders)
    if low_memory:
        stages = [gemma + connector + _LTX_PROMPT_WORKING_GIB, transformer + denoise_work, decode]
        if upsampler_file is not None:
            stages.append(transformer + upsampler + _LTX_UPSAMPLE_WORKING_GIB)
        peak = max(stages)
    else:
        peak = gemma + connector + transformer + upsampler + max(denoise_work, decode - _LTX_DECODE_RELEASED_GIB)
    return int(peak * GIB)


def clip_memory_fit(
    resolved_model: str,
    *,
    width: int,
    height: int,
    frames: int,
    low_memory: bool,
    upscale: bool = False,
    budget: MemoryBudget | None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
    env: Mapping[str, str] | None = None,
    model_name: str | None = None,
    app_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Check a clip against the memory budget; callers warn on ``tight`` or ``too_large`` and never block.

    Args:
        resolved_model: The resolved model reference (a local directory or a Hugging Face repo id).
        width: Output width in pixels, as aligned for the job.
        height: Output height in pixels, as aligned for the job.
        frames: Output frame count, as aligned for the job (an extension window carries its carried frames).
        low_memory: Whether the pipeline frees Gemma before loading the transformer.
        upscale: Whether the clip is upscaled 2x.
        budget: The accelerator's memory budget, or ``None`` when unknown.
        find_local_dir: Locates a downloaded model directory.
        env: The environment to read the decode budget from; the process environment when omitted.
        model_name: The name the model was picked by (``ltx-4``), which sets the quantize level on a discrete GPU.
        app_config: The loaded config, which sets that level and names the single-file transformer (discrete GPU only).

    Returns:
        ``{"status", "required_gb", "budget_gb"}``, or ``None`` when no memory budget is known or the model is not a
        downloaded LTX model with its text encoder. On a discrete GPU (CUDA) it adds ``"kind": "discrete"`` and
        ``"resource"`` (``"gpu"`` or ``"system"``): the memory the numbers are about, whichever is closer to its limit.
    """
    if budget is None or not budget.gpu_bytes:
        return None
    if budget.system_bytes is not None:
        return _cuda_clip_fit(
            resolved_model,
            width=width,
            height=height,
            frames=frames,
            low_memory=low_memory,
            upscale=upscale,
            budget=budget,
            find_local_dir=find_local_dir,
            model_name=model_name,
            app_config=app_config,
        )
    model_dir = find_local_dir(resolved_model)
    if model_dir is None or not is_ltx_mlx_layout(model_dir):
        return None
    text_encoder_dir = find_local_dir(ltx_mlx_text_encoder_repo())
    if text_encoder_dir is None:
        return None
    required = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, width=width, height=height, frames=frames, low_memory=low_memory, upscale=upscale, unified_bytes=budget.unified_bytes, env=env)
    if required is None:
        return None
    return {"status": classify_memory_fit(required, budget.gpu_bytes), "required_gb": round(required / GIB, 1), "budget_gb": round(budget.gpu_bytes / GIB, 1)}


def ltx_cuda_system_memory(
    resolved_model: str,
    model_name: str | None,
    app_config: dict[str, Any] | None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> LtxCudaMemoryEstimate | None:
    """Return the system memory of the LTX model picked as *model_name* on a CUDA machine, or ``None`` when its level or files are unknown.

    The level comes from the config (``ltx-4`` is 4). Nothing is downloaded: the files must already be on disk.
    """
    from zvisiongenerator.backends import estimate_ltx_system_memory
    from zvisiongenerator.utils.config import ltx_diffusers_quantize

    config = app_config or {}
    quantize = ltx_diffusers_quantize(config, model_name)
    if quantize is None:
        return None
    return estimate_ltx_system_memory(config, resolved_model, quantize, find_local_dir=find_local_dir)


def _cuda_clip_fit(
    resolved_model: str,
    *,
    width: int,
    height: int,
    frames: int,
    low_memory: bool,
    upscale: bool,
    budget: MemoryBudget,
    find_local_dir: Callable[[str], Path | None],
    model_name: str | None,
    app_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Check an LTX clip against a discrete GPU's budget, or return ``None`` when the model's files or level are unknown."""
    system = ltx_cuda_system_memory(resolved_model, model_name, app_config, find_local_dir)
    if system is None:
        return None
    estimate = estimate_ltx_cuda_clip_memory(system, width=width, height=height, frames=frames, low_memory=low_memory, upscale=upscale)
    # Report the memory that is closer to its limit, so the note's numbers explain its status.
    system_budget = budget.system_bytes or 0
    resource, required, limit = "gpu", estimate.gpu_bytes, budget.gpu_bytes
    if system_budget and estimate.system_bytes / (system_budget * _CUDA_SYSTEM_FITS_RATIO) > estimate.gpu_bytes / (budget.gpu_bytes * _CUDA_GPU_FITS_RATIO):
        resource, required, limit = "system", estimate.system_bytes, system_budget
    status = classify_ltx_cuda_memory_fit(estimate, budget)
    # A clip out of GPU memory is about the GPU, even when system memory is fuller.
    if status == "too_large" and estimate.gpu_bytes > budget.gpu_bytes:
        resource, required, limit = "gpu", estimate.gpu_bytes, budget.gpu_bytes
    return {"status": status, "required_gb": round(required / GIB, 1), "budget_gb": round(limit / GIB, 1), "kind": "discrete", "resource": resource}


def classify_memory_fit(required_bytes: int, budget_bytes: int) -> str:
    """Classify an estimate against the memory budget as ``fits``, ``tight``, or ``too_large``.

    ``fits`` stays within 1.1x Apple's recommended working set; ``tight`` stays within MLX's default memory
    limit (1.5x it), where the model runs but macOS compresses and swaps other memory; beyond is ``too_large``.
    """
    if required_bytes <= budget_bytes * _FITS_RATIO:
        return "fits"
    if required_bytes <= budget_bytes * _TIGHT_LIMIT_RATIO:
        return "tight"
    return "too_large"


@dataclass(frozen=True)
class LtxCudaMemoryEstimate:
    """Estimated system memory of LTX on the diffusers/CUDA backend, in bytes, for its two ways of holding the model."""

    text_encoder_bytes: int  # the text encoder while it encodes a prompt
    transformer_bytes: int  # the transformer, held while generating
    resident_bytes: int = 0  # the connectors, held throughout
    quantize: int = 4  # the transformer's level, 4 or 8

    @property
    def taking_turns(self) -> int:
        """The peak when the transformer is released while the text encoder runs."""
        return max(self.text_encoder_bytes, self.transformer_bytes) + self.resident_bytes + _LTX_CUDA_RUNTIME_BYTES

    @property
    def together(self) -> int:
        """The peak when the transformer stays loaded while the text encoder runs."""
        return self.text_encoder_bytes + self.transformer_bytes + self.resident_bytes + _LTX_CUDA_RUNTIME_BYTES


def estimate_ltx_cuda_system_memory(*, block_matrices: int, transformer_rest: int, connectors: int, text_encoder: int, quantize: int) -> LtxCudaMemoryEstimate:
    """Estimate the system memory of LTX on the diffusers/CUDA backend.

    At q4 the transformer blocks' 2-D weights are NF4 and the rest stays bfloat16; at q8 they are FP8. Both levels
    keep the connectors in FP8 (one byte per element) and the text encoder mostly in FP8.

    Args:
        block_matrices: Elements of the 2-D weights inside the transformer blocks (what q4 packs).
        transformer_rest: The transformer's other elements.
        connectors: Elements of the text connectors, which are held throughout.
        text_encoder: Bytes of the FP8 text encoder (its embeddings and norms stay bfloat16).
        quantize: 4 or 8.
    """
    bytes_per_block_weight = {8: 1, 4: _NF4_BYTES_PER_WEIGHT}[quantize]
    transformer_bytes = block_matrices * bytes_per_block_weight + transformer_rest * 2
    return LtxCudaMemoryEstimate(text_encoder_bytes=text_encoder, transformer_bytes=int(transformer_bytes), resident_bytes=connectors, quantize=quantize)


def estimate_ltx_cuda_clip_memory(
    system: LtxCudaMemoryEstimate,
    *,
    width: int,
    height: int,
    frames: int,
    low_memory: bool,
    upscale: bool = False,
) -> CudaMemoryEstimate:
    """Estimate the peak GPU and system memory of one LTX clip on the diffusers/CUDA backend.

    In low-memory mode the transformer streams to the GPU block by block and the peak depends only on the clip's
    token count. Without it the whole transformer stays on the GPU as well. System memory is the weights' peak plus,
    for the two-stage upscale, the second stage's extra components.

    Args:
        system: The weights' system memory, from :func:`estimate_ltx_cuda_system_memory`.
        width: Output width in pixels.
        height: Output height in pixels.
        frames: Output frame count.
        low_memory: Whether the transformer streams to the GPU instead of staying there.
        upscale: Whether the clip is upscaled 2x.
    """
    tokens = ltx_video_tokens(width, height, frames)
    plateau = _LTX_CUDA_GPU_PLATEAU_BYTES[system.quantize]
    gpu = plateau + max(tokens - _LTX_CUDA_GPU_KNEE_TOKENS[system.quantize], 0) * _LTX_CUDA_GPU_BYTES_PER_TOKEN
    if upscale:
        gpu = max(gpu, plateau + _LTX_CUDA_UPSCALE_GPU_BYTES)
    gpu += _LTX_CUDA_GPU_HEADROOM_BYTES
    if not low_memory:
        gpu += system.transformer_bytes
    return CudaMemoryEstimate(gpu_bytes=int(gpu), system_bytes=system.taking_turns + (_LTX_CUDA_UPSCALE_SYSTEM_BYTES if upscale else 0))


def classify_cuda_memory_fit(estimate: CudaMemoryEstimate, budget: MemoryBudget) -> str:
    """Classify a CUDA estimate as ``fits``, ``tight`` (runs, but slowly), or ``too_large`` (out of GPU memory).

    ``too_large`` needs more GPU memory than the card has. ``tight`` needs over 90% of it, or weights over 80%
    of system memory, so part of them streams from disk; otherwise it ``fits``.
    """
    if estimate.gpu_bytes > budget.gpu_bytes:
        return "too_large"
    system_limit = (budget.system_bytes or 0) * _CUDA_SYSTEM_FITS_RATIO
    if estimate.gpu_bytes > budget.gpu_bytes * _CUDA_GPU_FITS_RATIO or estimate.system_bytes > system_limit:
        return "tight"
    return "fits"


def classify_ltx_cuda_memory_fit(estimate: CudaMemoryEstimate, budget: MemoryBudget) -> str:
    """Classify an LTX estimate on CUDA like :func:`classify_cuda_memory_fit`, but weights beyond system memory are ``too_large``.

    The LTX backend refuses to load weights that need more than the machine's system memory, unlike an image model.
    """
    if budget.system_bytes and estimate.system_bytes > budget.system_bytes:
        return "too_large"
    return classify_cuda_memory_fit(estimate, budget)


def _cuda_level_estimate(components: dict[str, WeightTotals], quantize: int | None, text_encoder_offloaded: bool) -> CudaMemoryEstimate:
    vae = components["vae"].bfloat16_bytes if "vae" in components else 0
    unquantized = sum(totals.bfloat16_bytes for name, totals in components.items() if name not in ("vae", *_CUDA_QUANTIZED_COMPONENTS))
    if quantize == 4:
        nf4 = {name: _cuda_component_bytes(components[name], _NF4_BYTES_PER_WEIGHT) for name in _CUDA_QUANTIZED_COMPONENTS if name in components}
        total = sum(nf4.values())
        on_gpu = max(nf4.values(), default=0) if text_encoder_offloaded else total
        return CudaMemoryEstimate(int(on_gpu + vae + _CUDA_RESIDENT_WORKING_BYTES), int(total - on_gpu + unquantized + _CUDA_SYSTEM_WORKING_BYTES))
    bytes_per_weight = 1 if quantize == 8 else 2
    streamed = sum(_cuda_component_bytes(components[name], bytes_per_weight) for name in _CUDA_QUANTIZED_COMPONENTS if name in components)
    return CudaMemoryEstimate(vae + _CUDA_STREAMED_GPU_BYTES, int(streamed + unquantized + _CUDA_SYSTEM_WORKING_BYTES))


def _cuda_ideogram4_estimate(components: dict[str, WeightTotals], quantize: int | None) -> CudaMemoryEstimate:
    """Estimate Ideogram 4 Instant on CUDA, mirroring ``backends/cuda/image/ideogram4._place``.

    The transformer stays in system memory at every level (NF4, FP8 or bfloat16) and streams to the GPU block by
    block. The text encoder is stored NF4 and is counted as stored; it is on the GPU only while it encodes the
    prompt, before the transformer runs, so the GPU holds the larger of the two phases plus the VAE.
    """
    vae = components["vae"].bfloat16_bytes if "vae" in components else 0
    text_encoder = components["text_encoder"].bfloat16_bytes if "text_encoder" in components else 0
    rest = sum(totals.bfloat16_bytes for name, totals in components.items() if name not in ("vae", "text_encoder", "transformer"))
    bytes_per_weight = {4: _NF4_BYTES_PER_WEIGHT, 8: 1}.get(quantize, 2)
    transformer = _cuda_component_bytes(components["transformer"], bytes_per_weight) if "transformer" in components else 0
    gpu = max(text_encoder + _CUDA_RESIDENT_WORKING_BYTES, _CUDA_STREAMED_GPU_BYTES) + vae
    return CudaMemoryEstimate(int(gpu), int(transformer + text_encoder + rest + _CUDA_SYSTEM_WORKING_BYTES))


def _cuda_component_bytes(totals: WeightTotals, bytes_per_weight: float) -> float:
    """Return a component's bytes with its 2-D weights at *bytes_per_weight* and other floats in bfloat16."""
    return totals.float_matrix_elements * bytes_per_weight + totals.float_other_elements * 2 + totals.fp8_matrix_elements + totals.packed_bytes


def _image_components(model_dir: Path) -> dict[str, WeightTotals] | None:
    components: dict[str, WeightTotals] = {}
    for path in model_weight_files(model_dir):
        component = component_of(model_dir, path)
        skipped = _TEXT_ENCODER_SKIPPED_PREFIXES if component == "text_encoder" else ()
        try:
            components[component] = components.get(component, WeightTotals()) + read_safetensors_totals(path, skipped)
        except OSError, ValueError, KeyError, TypeError:
            return None
    return components or None


def _component_quantize(name: str, quantize: int | None, quantize_text_encoder: bool) -> int | None:
    """Return the level *name* is quantized at: none for a text encoder the loader keeps unquantized."""
    return None if name == "text_encoder" and not quantize_text_encoder else quantize


def _image_component_bytes(name: str, totals: WeightTotals, quantize: int | None) -> float:
    if name == "vae":
        return (totals.float_matrix_elements + totals.float_other_elements) * 4 + totals.packed_bytes
    if quantize and not totals.prequantized:
        matrices = totals.float_matrix_elements + totals.fp8_matrix_elements
        return matrices * (quantize / 8 + _QUANT_GROUP_OVERHEAD_BYTES) + totals.float_other_elements * 2 + totals.packed_bytes
    return totals.bfloat16_bytes


def _files_bfloat16_bytes(paths: list[Path] | tuple[Path, ...]) -> int:
    """Return bytes resident when *paths* load with floats as bfloat16 and packed weights as stored."""
    return sum((read_safetensors_totals(path) for path in paths if path.is_file()), WeightTotals()).bfloat16_bytes
