"""Size the weights of a model and estimate an LTX clip's memory: unified memory on MLX, GPU and system memory on CUDA.

Sizes read only safetensors headers (tensor dtypes and shapes), never the weights, and mirror how each backend holds each
component in memory. The image estimates live in :mod:`zvisiongenerator.utils.memory_verdict` and the MiniMax H3 clip estimates in
:mod:`zvisiongenerator.utils.h3_memory`; this module holds what they are built from (the weight totals and the memory budget) and the LTX
clip estimates, whose working-memory curves were measured over the clip's size and length.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.model_files import (
    component_of,
    find_local_model_dir,
    is_ltx_mlx_layout,
    is_moss_model,
    ltx_mlx_bundled_text_encoder,
    ltx_mlx_transformer_file,
    ltx_mlx_upsampler_file,
    ltx_mlx_video_decoder_file,
    model_weight_files,
)
from zvisiongenerator.utils.moss_build import moss_base_dir, turbo_lora_file
from zvisiongenerator.utils.safetensors_stream import read_header

GIB = 1024**3
# The generic MLX working memory above an image model's weights at 1 MP, which a model with no measured working memory of its own reads: the middle of the 1 MP working memory of Z-Image, Krea 2 and Klein at q4 and q8,
# fitted to the process's peak footprint. not measured for a model with no row; the per-model terms of ``memory_verdict`` correct it where a model was measured.
IMAGE_WORKING_BYTES = int(1.2 * GIB)
# Ideogram 4 Instant on MLX, above its weights at 1024x1024: the q8 figure, to which ``memory_verdict`` adds each level's distance. measured: ideo-m, ideo-q4-m, ideo-none-m.
_IDEOGRAM4_MLX_WORKING_BYTES = int(2.79 * GIB)
# bitsandbytes NF4 packs two weights per byte plus a double-quantized scale per 64-weight block (~4.13 bits).
_NF4_BYTES_PER_WEIGHT = 4.13 / 8
# LTX on CUDA: beyond the weights, the process holds the CUDA runtime, the VAEs' and tokenizer's CPU side and the copies a load stages, per level. The figure is the process's memory (anonymous plus shared pages, the highest the two summed to in
# one sample) less the weights taking turns. The shared pages are the pinned copies of the streamed blocks, which are real memory the machine loses and cannot swap; they are not counted twice, the kernel keeps RssAnon and RssShmem apart.
# measured: ltx8-s (q8); ltx4-s on LTX-2.3 and LTX-2.5 (q4, the mean of the two).
_LTX_CUDA_RUNTIME_BYTES = {4: int(5.04 * GIB), 8: int(3.08 * GIB)}
# LTX on CUDA, per clip: the GPU need of the streamed transformer (low-memory mode), before the CUDA offset ``memory_verdict`` adds once. It is flat up to a knee, then grows linearly with the token count (the float32 temporaries outside the
# chunked layers). Each term is its probed figure less the offset: the middle of the case's hold-probe bracket (the allocator of a streaming LTX job holds 1-2 GiB more than its allocated peak and gives it back under a hold, so the peak alone
# understates it). The slope is a line through the three long q4 clips, and q8 shares it (the allocated peaks agree to 1 %). measured: ltx4-s, ltx4-m, ltx4-xl-10s, ltx4-xl-14s, ltx4-xl-15s (q4, up to 64.4k tokens); ltx8-s, ltx8-xl-10s (q8, up to 43.4k tokens); extrapolated beyond.
_LTX_CUDA_GPU_PLATEAU_BYTES = {4: int(4.36 * GIB), 8: int(3.59 * GIB)}
_LTX_CUDA_GPU_KNEE_TOKENS = {4: 25200, 8: 16600}
_LTX_CUDA_GPU_BYTES_PER_TOKEN = int(1.07e-4 * GIB)
# The two-stage upscale also holds the latent upsampler on the GPU: the figure is the allocated peak of a small clip above the plateau term, not a probed one. measured at q4: ltx4-s-upscale, ltx4-l-upscale; not measured at q8 (carried unchanged).
_LTX_CUDA_UPSCALE_GPU_BYTES = int(0.68 * GIB)
# The two-stage upscale also holds the upsampler and the first stage's latents, in system memory above the plain clip. measured at q4: ltx4-s-upscale; not measured at q8.
_LTX_CUDA_UPSCALE_SYSTEM_BYTES = int(0.93 * GIB)

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
# MOSS-Image on MLX: the peak footprint does not rise with the size, so the text encoder stage (7.85 GiB of weights) sets it, and its working memory is the middle of the two rows. measured: moss-m, moss-xl.
_MOSS_MLX_TEXT_ENCODER_WORKING_BYTES = int(1.0 * GIB)
# The denoiser stage's working memory above its weights (the loop and the strip VAE decode): the rows give only the larger stage, so this stays under the bound they set. not measured: no footprint row isolates it.
_MOSS_MLX_DENOISE_WORKING_BYTES = int(1.9 * GIB)
# mflux loads only MOSS-Image's language model; the Qwen3.5 vision tower (``visual.``) stays on disk, and a stored copy has none.
_MOSS_MLX_TEXT_ENCODER_SKIPPED_PREFIXES = (*_TEXT_ENCODER_SKIPPED_PREFIXES, "visual.")
_LTX_MLX_CONNECTOR_FILE = "connector.safetensors"
# The audio decoders loaded for a decode next to the video VAE decoder: the VAE tiler's budget limits only activations, not these weights.
_LTX_MLX_AUDIO_DECODER_FILES = ("audio_vae.safetensors", "vocoder.safetensors")
# LTX MLX working memory beyond the weights, in GiB, fitted to the process's peak footprint (a lifetime peak of ``phys_footprint``, which the MLX buffer cache does not inflate: the app runs the generation with the cache off) where the
# transformer is not streamed; Macs with 24 GiB or less stream it, so the estimate is conservative there (not measured). Each term is an absolute curve, not scaled by model.
#
# The load phase (the prompt stage: Gemma and the connector held while the prompt is encoded) sets the peak of a clip whose denoise stage is smaller: it is flat across the small ltx-4 sizes and with an upscale, above Gemma 3 and the connector.
# LTX-2.5 (Gemma 4, bundled in its pack) has its own figure. measured: ltx4-s, ltx4-m, ltx4-s-upscale (ltx-4), ltx25-4-s (LTX-2.5).
_LTX_PROMPT_WORKING_GIB = 3.07
_LTX_PROMPT_WORKING_BUNDLED_GIB = 4.27
# The denoise stage grows with the video tokens, the line through two rows above ltx-8's transformer. measured: ltx8-s, ltx8-m. A smaller transformer (ltx-4) never reaches it in a measured clip, so the load phase sets its peak.
_LTX_DENOISE_AT_REFERENCE_GIB = 3.41
_LTX_REFERENCE_TOKENS = 5376  # 768x448 at 121 frames.
_LTX_DENOISE_GIB_PER_TOKEN = 0.000247
# Image to video (``i2v``) and Extend add the conditioning's own encode to the peak. measured: ltx4-s-i2v, ltx4-s-extend; the same constant carries to other sizes and packs (not measured there).
_LTX_MODE_GIB = {"i2v": 0.56, "extend": 4.20}
# The decode stage and the upsample stage below are fitted to MLX's peak, not measured on the footprint: no footprint row isolates them (the load or the denoise phase sets every measured peak).
_LTX_DECODE_BASE_GIB = 12.9  # Peak of a VAE decode (decoder weights included) up to _LTX_DECODE_KNEE_TOKENS tokens,
_LTX_DECODE_KNEE_TOKENS = 2048
_LTX_DECODE_GIB_PER_TOKEN = 0.00067  # ... growing by this per token beyond it, until the tiler's budget caps it.
_LTX_UPSAMPLE_WORKING_GIB = 0.5  # Transformer + spatial upsampler + this, before the second denoise stage.
_LTX_DECODE_RELEASED_GIB = 2.83  # With Low memory off, the decode peak includes weights that stay resident anyway.
_LTX_VAE_DECODE_BUDGET_ENV = "LTX2_VAE_DECODE_BUDGET_GB"  # Mirrors ltx_core_mlx's decode_budget_bytes().
# The streaming path (Macs with 24 GiB or less, Low memory on) holds the process's peak footprint flat whatever the job: the peak is set while the text encoder loads (see ``_LOW_RAM_THRESHOLD_BYTES`` in ``backends/mac/video.py``), and the Gemma 3 packs
# share that encoder and connector, so ``ltx-8`` reads ``ltx-4``'s constant. The transformer is not added: the figure does not move with the job's size. User LoRAs are not counted on this path (not measured). measured, forced onto the path
# with ``--total-gib``: ltx4-s, ltx4-s-i2v, ltx4-s-extend, ltx4-m, ltx4-s-upscale, ltx25-4-s (the bundled figure, t2v only), ltx8-s.
_LTX_STREAMING_GIB = 14.77
_LTX_STREAMING_BUNDLED_GIB = 15.60
_LTX_TOKEN_FRAME_STEP = 8
_LTX_TOKEN_PIXELS = 32


@dataclass(frozen=True)
class MemoryBudget:
    """The memory a model may use on this machine.

    On Apple Silicon (unified memory) *gpu_bytes* is Apple's recommended GPU working set, *system_bytes* is
    ``None`` and *unified_bytes* is the total unified memory, which a Mac verdict is graded against. On a discrete GPU (CUDA)
    *gpu_bytes* is the card's budget (its total less the driver's reserve) and *system_bytes* the system memory that
    offloaded weights stream from. *gpu_total_bytes* is the card's whole memory, reserve included, for the features tuned on
    that size (SeedVR2's tier, the Smart prompt enhancer, the first-run model pick); the memory engine reads *gpu_bytes*.
    """

    gpu_bytes: int
    system_bytes: int | None = None
    unified_bytes: int | None = None
    gpu_total_bytes: int | None = None

    @property
    def full_gpu_bytes(self) -> int:
        """The card's whole memory, or *gpu_bytes* where there is no separate total (a Mac's working set, a card whose size could not be read)."""
        return self.gpu_total_bytes or self.gpu_bytes


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
    embedding_elements: int = 0  # token-embedding tables (``embed_tokens``), already counted in float_matrix_elements

    def __add__(self, other: WeightTotals) -> WeightTotals:
        return WeightTotals(
            self.float_matrix_elements + other.float_matrix_elements,
            self.float_other_elements + other.float_other_elements,
            self.packed_bytes + other.packed_bytes,
            self.prequantized or other.prequantized,
            self.fp8_matrix_elements + other.fp8_matrix_elements,
            self.embedding_elements + other.embedding_elements,
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
    _, header = read_header(Path(path))
    return totals_from_header(header, skipped_prefixes, label=path)


def totals_from_header(header: Mapping[str, Mapping[str, Any]], skipped_prefixes: tuple[str, ...] = (), *, label: str = "the header") -> WeightTotals:
    """Summarise a safetensors *header* (tensor name to ``{"dtype", "shape", ...}``, without its metadata entry) by how the loaders hold the tensors.

    Args:
        header: The tensors of one or more files; only each one's dtype and shape are read.
        skipped_prefixes: Tensor-name prefixes the loader never reads, left out of the totals.
        label: What the header is of, named in the error.

    Raises:
        ValueError: If a tensor uses a dtype the estimate does not know.
    """
    matrix = other = packed = fp8_matrix = embedding = 0
    prequantized = False
    for name, spec in header.items():
        if name.startswith(skipped_prefixes):
            continue
        dtype = str(spec["dtype"])
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} in {label}")
        shape = [int(dim) for dim in spec["shape"]]
        elements = math.prod(shape)
        if dtype.startswith("F8_") and len(shape) == 2:
            fp8_matrix += elements
        elif dtype not in _FLOAT_DTYPES:
            packed += elements * _DTYPE_BYTES[dtype]
            prequantized = prequantized or dtype == "U32"
        elif len(shape) == 2:
            matrix += elements
            if "embed_tokens" in name:
                embedding += elements
        else:
            other += elements
    return WeightTotals(matrix, other, packed, prequantized, fp8_matrix, embedding)


def mlx_image_bytes(components: dict[str, WeightTotals], quantize: int | None, *, family: str | None, lora_bytes: int = 0) -> int:
    """Estimate resident memory of an mflux image model from its component totals, at one quantize level.

    Args:
        components: Each component's totals, as :func:`_image_components` reads them (MOSS-Image's text encoder without its vision tower).
        quantize: ``None`` for unquantized, or 4 / 8 bits.
        family: The image family; ``"moss"`` loads its stages one after the other, ``"ideogram4"`` adds its measured working memory and ``"krea2"`` keeps its text encoder in bfloat16.
        lora_bytes: MOSS-Image's Turbo LoRA bytes plus any user LoRAs, which mflux keeps as separate layers; ignored for other families, whose LoRAs mflux merges into the weights.

    Returns:
        Estimated peak bytes, for 1 MP and no references.
    """
    if family == "moss":
        return _mlx_moss_estimate(components, quantize, lora_bytes)
    # mflux keeps Krea 2's text encoder in bfloat16 at every quantize level.
    quantize_text_encoder = family != "krea2"
    working = _IDEOGRAM4_MLX_WORKING_BYTES if family == "ideogram4" else IMAGE_WORKING_BYTES
    return int(sum(image_component_bytes(name, totals, _component_quantize(name, quantize, quantize_text_encoder)) for name, totals in components.items())) + working


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


@dataclass(frozen=True)
class LtxMlxWeights:
    """The bfloat16 bytes of the components of an LTX MLX pack, as the memory estimate reads them from the safetensors headers."""

    gemma: int  # the text encoder: the pack's bundled one, else the Gemma 3 repo
    connector: int
    transformer: int
    decoders: int  # the video decoder the pipeline picks for the pack, the audio VAE and the vocoder
    upsampler: int | None = None  # the spatial upsampler, ``None`` when the clip is not upscaled or the pack has none
    bundled_encoder: bool = False  # whether the pack carries its own text encoder (LTX-2.5's Gemma 4) rather than using the Gemma 3 repo


def ltx_mlx_weights(model_dir: Path, text_encoder_dir: Path | None, *, upscale: bool = True, file_bytes: Callable[[Path], int] | None = None) -> LtxMlxWeights | None:
    """Return the bfloat16 bytes of the components of the LTX MLX pack in *model_dir*, read from the safetensors headers.

    The text encoder *G* is the pack's bundled one (LTX-2.5) when it has one, else the Gemma 3 in *text_encoder_dir*; the
    decoders *D* are the video decoder the pipeline picks for the pack, the audio VAE and the vocoder.

    Args:
        model_dir: The LTX MLX model directory.
        text_encoder_dir: The Gemma 3 text-encoder directory; ``None`` is accepted for a pack with a bundled text encoder.
        upscale: Whether to include the spatial upsampler; without an upsampler file in *model_dir* it is left out either way.
        file_bytes: Returns the bfloat16 bytes of one weight file; reads its safetensors header when omitted. A caller that
            knows the sizes without the files (a folder of empty stand-ins named like the pack's files) passes its own.

    Returns:
        The components' bytes, or ``None`` when the layout is not LTX MLX, the pack has no bundled text encoder and *text_encoder_dir* is ``None``, or weights are unreadable.
    """
    transformer_file = ltx_mlx_transformer_file(model_dir)
    if not is_ltx_mlx_layout(model_dir) or transformer_file is None:
        return None
    bundled_encoder = ltx_mlx_bundled_text_encoder(model_dir)
    if bundled_encoder is None and text_encoder_dir is None:
        return None
    encoder_files = [bundled_encoder] if bundled_encoder is not None else model_weight_files(text_encoder_dir)
    upsampler_file = ltx_mlx_upsampler_file(model_dir) if upscale else None
    video_decoder = ltx_mlx_video_decoder_file(model_dir)
    decoder_files = [*([video_decoder] if video_decoder is not None else []), *(model_dir / name for name in _LTX_MLX_AUDIO_DECODER_FILES)]
    measure = file_bytes or _file_bfloat16_bytes

    def total(paths: list[Path]) -> int:
        return sum(measure(path) for path in paths if path.is_file())

    try:
        return LtxMlxWeights(
            gemma=total(encoder_files),
            connector=total([model_dir / _LTX_MLX_CONNECTOR_FILE]),
            transformer=total([transformer_file]),
            decoders=total(decoder_files),
            upsampler=total([upsampler_file]) if upsampler_file is not None else None,
            bundled_encoder=bundled_encoder is not None,
        )
    except OSError, ValueError, KeyError, TypeError:
        return None


def _streams(low_memory: bool, unified_bytes: int | None) -> bool:
    """Return whether the Mac backend streams the transformer from disk, by the backend's own rule."""
    from zvisiongenerator.backends.mac.video import _use_low_ram_streaming  # lazy: the backend package imports this module

    return _use_low_ram_streaming(low_memory, unified_bytes)


def ltx_mlx_clip_bytes(
    weights: LtxMlxWeights,
    *,
    width: int,
    height: int,
    frames: int,
    low_memory: bool,
    unified_bytes: int | None = None,
    env: Mapping[str, str] | None = None,
    lora_bytes: int = 0,
    mode: str = "t2v",
) -> int:
    """Estimate peak memory of an LTX MLX clip from its component *weights* (as :func:`ltx_mlx_weights` reads them).

    The peak is the largest stage. The weights are the bfloat16 bytes of the text encoder *G*, connector *C*, transformer *T*,
    spatial upsampler *U* and decoders *D* (video VAE, audio VAE, vocoder). The working memory of each stage is a curve
    (the ``_LTX_*_GIB`` constants) over the video tokens of the final output size, which is stage 2 when upscaling::

        prompt   = G + C + prompt working memory                # the pack's own: Gemma 3, or LTX-2.5's bundled Gemma 4
        denoise  = T + denoise working memory(tokens)
        decode   = min(decode budget + D, decode peak(tokens))   # the decode peak includes D
        upsample = T + U + upsample working memory              # upscale only

    With ``low_memory`` the stages run one after another and the peak is their maximum. Without it Gemma, the
    connector, the transformer and the upsampler stay resident, so their weights add to the larger of the
    denoise working memory and the decode peak less the weights it already counts. The decode budget is the VAE
    tiler's (half of unified memory, or ``LTX2_VAE_DECODE_BUDGET_GB``); it limits activations, not weights.

    An image-to-video clip and an Extend add their conditioning's encode to the peak (``mode`` ``"i2v"`` and ``"extend"``). The constants are
    fit to the process's peak footprint where the transformer is not streamed. Macs with 24 GiB or less stream the transformer when ``low_memory`` is on, and there the
    peak is a constant per pack (``_LTX_STREAMING_GIB``, ``_LTX_STREAMING_BUNDLED_GIB``), whatever the clip. It ignores memory used by other apps.

    Args:
        weights: The pack's component bytes; a spatial upsampler stage counts only when ``weights.upsampler`` is set.
        width: Output width in pixels, as aligned for the job.
        height: Output height in pixels, as aligned for the job.
        frames: Output frame count, as aligned for the job.
        low_memory: Whether the pipeline frees Gemma before loading the transformer.
        unified_bytes: The machine's unified memory; without it the decode peak is not capped.
        env: The environment to read the decode budget from; the process environment when omitted.
        mode: ``"t2v"``, ``"i2v"`` (the clip starts from a picture) or ``"extend"`` (it continues a source video).
        lora_bytes: The total size of the user LoRA files. The backend attaches them unfused (``_DEFAULT_LORA_MODE`` in ``backends/mac/video.py``), and
            ltx_core_mlx's ``attach_loras`` adds each file's ``lora_a`` and ``lora_b`` to the transformer's own layers, so they count with the transformer.

    Returns:
        Estimated peak bytes.
    """
    if _streams(low_memory, unified_bytes):
        return int((_LTX_STREAMING_BUNDLED_GIB if weights.bundled_encoder else _LTX_STREAMING_GIB) * GIB)
    gemma, connector, transformer, decoders = weights.gemma / GIB, weights.connector / GIB, (weights.transformer + lora_bytes) / GIB, weights.decoders / GIB
    upsampler = (weights.upsampler or 0) / GIB
    tokens = ltx_video_tokens(width, height, frames)
    denoise_work = _LTX_DENOISE_AT_REFERENCE_GIB + _LTX_DENOISE_GIB_PER_TOKEN * (tokens - _LTX_REFERENCE_TOKENS)
    prompt_work = _LTX_PROMPT_WORKING_BUNDLED_GIB if weights.bundled_encoder else _LTX_PROMPT_WORKING_GIB
    decode = _LTX_DECODE_BASE_GIB + _LTX_DECODE_GIB_PER_TOKEN * max(0, tokens - _LTX_DECODE_KNEE_TOKENS)
    decode_budget = ltx_decode_budget_bytes(unified_bytes, os.environ if env is None else env)
    if decode_budget is not None:
        decode = min(decode, decode_budget / GIB + decoders)
    if low_memory:
        stages = [gemma + connector + prompt_work, transformer + denoise_work, decode]
        if weights.upsampler is not None:
            stages.append(transformer + upsampler + _LTX_UPSAMPLE_WORKING_GIB)
        peak = max(stages)
    else:
        peak = gemma + connector + transformer + upsampler + max(denoise_work, decode - _LTX_DECODE_RELEASED_GIB)
    return int((peak + _LTX_MODE_GIB.get(mode, 0.0)) * GIB)


def ltx_cuda_system_memory(
    resolved_model: str,
    model_name: str | None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> LtxCudaMemoryEstimate | None:
    """Return the system memory of the LTX model picked as *model_name* on a CUDA machine, or ``None`` when its level or files are unknown.

    The level comes from the model's metadata entry (``ltx-4`` is 4), or from a converted folder's manifest. Nothing is downloaded:
    the files must already be on disk.
    """
    from zvisiongenerator.backends import estimate_ltx_system_memory
    from zvisiongenerator.utils.ltx_facts import ltx_diffusers_quantize
    from zvisiongenerator.utils.ltx_converted import read_converted_ltx

    quantize = ltx_diffusers_quantize(model_name)
    if quantize is None:
        # A converted folder (always a local folder) carries its own level.
        converted = read_converted_ltx(Path(resolved_model).expanduser())
        if converted is None:
            return None
        quantize = converted["bits"]
    return estimate_ltx_system_memory(resolved_model, quantize, find_local_dir=find_local_dir, model_name=model_name)


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
        return max(self.text_encoder_bytes, self.transformer_bytes) + self.resident_bytes + _LTX_CUDA_RUNTIME_BYTES[self.quantize]

    @property
    def together(self) -> int:
        """The peak when the transformer stays loaded while the text encoder runs."""
        return self.text_encoder_bytes + self.transformer_bytes + self.resident_bytes + _LTX_CUDA_RUNTIME_BYTES[self.quantize]


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
    lora_bytes: int = 0,
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
        lora_bytes: The total size of the user LoRA files. The backend loads them as peft adapters (``_apply_loras`` in ``backends/cuda/video/backend.py``)
            on the transformer, so they sit wherever its layers do: in system memory with it, and on the GPU too when the transformer stays there (no
            low-memory mode). Streamed, each block carries its own adapters to the GPU, a share too small to count.
    """
    system = replace(system, transformer_bytes=system.transformer_bytes + lora_bytes)
    tokens = ltx_video_tokens(width, height, frames)
    plateau = _LTX_CUDA_GPU_PLATEAU_BYTES[system.quantize]
    gpu = plateau + max(tokens - _LTX_CUDA_GPU_KNEE_TOKENS[system.quantize], 0) * _LTX_CUDA_GPU_BYTES_PER_TOKEN
    if upscale:
        gpu = max(gpu, plateau + _LTX_CUDA_UPSCALE_GPU_BYTES)
    if not low_memory:
        # Low memory off is not measured: the streamed plateau plus the whole transformer is a composite no bench case ran.
        gpu += system.transformer_bytes
    return CudaMemoryEstimate(gpu_bytes=int(gpu), system_bytes=system.taking_turns + (_LTX_CUDA_UPSCALE_SYSTEM_BYTES if upscale else 0))


def estimate_ltx_cuda_together_bytes(system: LtxCudaMemoryEstimate, *, upscale: bool, lora_bytes: int) -> int:
    """Return the system memory of an LTX job that keeps the transformer and the text encoder in memory together, with the user LoRAs and, for the two-stage upscale, its second stage."""
    loaded = replace(system, transformer_bytes=system.transformer_bytes + lora_bytes)
    return loaded.together + (_LTX_CUDA_UPSCALE_SYSTEM_BYTES if upscale else 0)


def _mlx_moss_estimate(components: dict[str, WeightTotals], quantize: int | None, lora_bytes: int) -> int:
    """Estimate MOSS-Image on MLX: the larger of its two stages, each with its working memory.

    The text encoder stage holds the text encoder alone; it is freed before the denoiser stage loads the transformer,
    the float32 VAE and the LoRAs, which mflux keeps as separate layers in the file's dtype (the Turbo LoRA is
    float32, so its file size is its footprint).
    """
    text_encoder = image_component_bytes("text_encoder", components["text_encoder"], quantize) if "text_encoder" in components else 0
    denoiser = sum(image_component_bytes(name, totals, quantize) for name, totals in components.items() if name != "text_encoder") + lora_bytes
    return int(max(text_encoder + _MOSS_MLX_TEXT_ENCODER_WORKING_BYTES, denoiser + _MOSS_MLX_DENOISE_WORKING_BYTES))


def _nf4_component_bytes(name: str, totals: WeightTotals) -> float:
    """Return a component's resident bytes under bitsandbytes NF4: 2-D weights in NF4, other floats in bfloat16.

    A text encoder's token-embedding table stays in bfloat16, because bitsandbytes quantizes only ``nn.Linear``.
    """
    embedding = totals.embedding_elements * (2 - _NF4_BYTES_PER_WEIGHT) if name == "text_encoder" else 0
    return _cuda_component_bytes(totals, _NF4_BYTES_PER_WEIGHT) + embedding


def level_component_bytes(name: str, totals: WeightTotals, level: int | None) -> float:
    """Return a CUDA image component's resident bytes at quantize *level*: NF4 (4), FP8 (8) or bfloat16 (``None``).

    The text encoder's token-embedding table stays in bfloat16 at both quantized levels: bitsandbytes quantizes only
    ``nn.Linear`` and :func:`~zvisiongenerator.backends.cuda.quant.cast_to_fp8` skips modules named ``embed``.

    Raises:
        ValueError: If *level* is not ``None``, 4 or 8.
    """
    if level is None:
        return _cuda_component_bytes(totals, 2)
    if level == 4:
        return _nf4_component_bytes(name, totals)
    if level == 8:
        embedding = totals.embedding_elements if name == "text_encoder" else 0  # one extra byte each over FP8's one
        return _cuda_component_bytes(totals, 1) + embedding
    raise ValueError(f"Unsupported quantize level {level!r}; expected None, 4 or 8.")


def turbo_lora_bytes(model_dir: Path) -> int:
    """Return the bytes the Turbo LoRA occupies in memory; zero when the model has none."""
    lora = turbo_lora_file(moss_base_dir(model_dir))
    if lora is None:
        return 0
    # The LoRA is float32 on disk and in memory, so the file size is its footprint.
    return lora.stat().st_size


def _cuda_component_bytes(totals: WeightTotals, bytes_per_weight: float) -> float:
    """Return a component's bytes with its 2-D weights at *bytes_per_weight* and other floats in bfloat16."""
    return totals.float_matrix_elements * bytes_per_weight + totals.float_other_elements * 2 + totals.fp8_matrix_elements + totals.packed_bytes


def read_image_components(model_dir: Path, *, unified: bool) -> dict[str, WeightTotals] | None:
    """Return each component's weight totals of the image model in *model_dir*, or ``None`` when its weights are missing or unreadable.

    Args:
        model_dir: The model directory.
        unified: Whether the totals are for a unified-memory (MLX) backend, which leaves MOSS-Image's vision tower out of its text encoder.
    """
    skipped = _MOSS_MLX_TEXT_ENCODER_SKIPPED_PREFIXES if unified and is_moss_model(model_dir) else _TEXT_ENCODER_SKIPPED_PREFIXES
    return _image_components(model_dir, skipped)


def image_component_totals(component: str, header: Mapping[str, Mapping[str, Any]], *, unified: bool) -> WeightTotals:
    """Summarise the *header* of a file of the image model's *component* the way :func:`read_image_components` does: the text encoder without its unused tensors.

    Args:
        component: The component's folder name, such as ``"text_encoder"``.
        header: The file's tensors (see :func:`totals_from_header`).
        unified: Whether the totals are for MOSS-Image on a unified-memory (MLX) backend, which leaves its vision tower (``visual.``) out of the text encoder.
    """
    if component != "text_encoder":
        return totals_from_header(header)
    return totals_from_header(header, _MOSS_MLX_TEXT_ENCODER_SKIPPED_PREFIXES if unified else _TEXT_ENCODER_SKIPPED_PREFIXES)


def _image_components(model_dir: Path, text_encoder_skipped: tuple[str, ...] = _TEXT_ENCODER_SKIPPED_PREFIXES) -> dict[str, WeightTotals] | None:
    components: dict[str, WeightTotals] = {}
    for path in model_weight_files(model_dir):
        component = component_of(model_dir, path)
        skipped = text_encoder_skipped if component == "text_encoder" else ()
        try:
            components[component] = components.get(component, WeightTotals()) + read_safetensors_totals(path, skipped)
        except OSError, ValueError, KeyError, TypeError:
            return None
    return components or None


def _component_quantize(name: str, quantize: int | None, quantize_text_encoder: bool) -> int | None:
    """Return the level *name* is quantized at: none for a text encoder the loader keeps unquantized."""
    return None if name == "text_encoder" and not quantize_text_encoder else quantize


def image_component_bytes(name: str, totals: WeightTotals, quantize: int | None) -> float:
    if name == "vae":
        return (totals.float_matrix_elements + totals.float_other_elements) * 4 + totals.packed_bytes
    if quantize and not totals.prequantized:
        matrices = totals.float_matrix_elements + totals.fp8_matrix_elements
        return matrices * (quantize / 8 + _QUANT_GROUP_OVERHEAD_BYTES) + totals.float_other_elements * 2 + totals.packed_bytes
    return totals.bfloat16_bytes


def _file_bfloat16_bytes(path: Path) -> int:
    """Return bytes resident when the weight file *path* loads with floats as bfloat16 and packed weights as stored."""
    return read_safetensors_totals(path).bfloat16_bytes
