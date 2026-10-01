"""Estimate how much unified memory a downloaded model needs on the MLX backends.

Estimates read only safetensors headers (tensor dtypes and shapes), never the weights, and mirror how
the MLX backends hold each component in memory. They are deliberately rough: a fixed working-memory
margin stands in for activations, which vary with resolution, frame count, and LoRAs.
"""

from __future__ import annotations

import functools
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.utils.model_files import component_of, is_ltx_mlx_layout, ltx_mlx_transformer_file, model_weight_files

GIB = 1024**3
_IMAGE_WORKING_BYTES = int(1.5 * GIB)
_VIDEO_WORKING_BYTES = 3 * GIB
_TIGHT_RATIO = 0.8
_MAX_HEADER_BYTES = 100 * 1024 * 1024

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
_LTX_MLX_CONNECTOR_FILE = "connector.safetensors"
_LTX_MLX_DECODER_FILES = ("vae_decoder.safetensors", "vae_encoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors")


@dataclass(frozen=True)
class WeightTotals:
    """Summarise the tensors of one or more safetensors files by how the loaders hold them."""

    float_matrix_elements: int = 0  # 2-D floating-point tensors: the weights quantization packs
    float_other_elements: int = 0  # other floating-point tensors (biases, norms, convolutions)
    packed_bytes: int = 0  # non-float tensors (FP8, packed quantized weights, ints), resident as stored
    prequantized: bool = False  # packed U32 weights: already quantized, so requested quantization is ignored

    def __add__(self, other: WeightTotals) -> WeightTotals:
        return WeightTotals(
            self.float_matrix_elements + other.float_matrix_elements,
            self.float_other_elements + other.float_other_elements,
            self.packed_bytes + other.packed_bytes,
            self.prequantized or other.prequantized,
        )

    @property
    def bfloat16_bytes(self) -> int:
        """Return resident bytes with floats loaded as bfloat16 and everything else as stored."""
        return (self.float_matrix_elements + self.float_other_elements) * 2 + self.packed_bytes


def read_safetensors_totals(path: Path) -> WeightTotals:
    """Summarise a safetensors file from its header, cached per file version.

    Raises:
        ValueError: If the file is not a readable safetensors file, or uses a dtype the estimate does not
            know (e.g. sub-byte formats): guessing its size could mislabel a model, so it is not estimated.
    """
    stat = path.stat()
    return _read_totals_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=512)
def _read_totals_cached(path: str, size: int, mtime_ns: int) -> WeightTotals:
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
    matrix = other = packed = 0
    prequantized = False
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = str(spec["dtype"])
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} in {path}")
        shape = [int(dim) for dim in spec["shape"]]
        elements = math.prod(shape)
        if dtype not in _FLOAT_DTYPES:
            packed += elements * _DTYPE_BYTES[dtype]
            prequantized = prequantized or dtype == "U32"
        elif len(shape) == 2:
            matrix += elements
        else:
            other += elements
    return WeightTotals(matrix, other, packed, prequantized)


def estimate_image_memory(model_dir: Path, quantize_levels: tuple[int | None, ...] = (None,)) -> dict[int | None, int] | None:
    """Estimate resident memory for an mflux image model loaded in bfloat16, at each quantize level.

    Mirrors ``MfluxBackend.load_model``: floating-point weights load as bfloat16, the VAE is upcast to
    float32, FP8 and pre-quantized weights stay as stored, and quantizing packs 2-D weights. Only the
    files the loader reads count (see :func:`model_weight_files`), and headers are read once for all levels.

    Args:
        model_dir: Directory holding the model's safetensors files.
        quantize_levels: Levels to estimate: ``None`` for unquantized, or 4 / 8 bits.

    Returns:
        Estimated peak bytes per level, or ``None`` when weights are missing or unreadable.
    """
    components = _image_components(model_dir)
    if components is None:
        return None
    return {level: int(sum(_image_component_bytes(name, totals, level) for name, totals in components.items())) + _IMAGE_WORKING_BYTES for level in quantize_levels}


def estimate_ltx_mlx_memory(model_dir: Path, text_encoder_dir: Path, *, low_memory: bool = True) -> int | None:
    """Estimate peak memory for ltx-pipelines-mlx.

    In ``low_memory`` mode (the default) the pipeline frees Gemma before loading the transformer and
    decoders, but the connector stays loaded alongside them, so the peak is the larger of Gemma + connector
    and connector + transformer + decoders. Without it every component stays loaded. The optional upscale
    pass (spatial upsampler plus a second denoise stage) is not included.

    Args:
        model_dir: The LTX MLX model directory.
        text_encoder_dir: The Gemma text-encoder directory.
        low_memory: Whether the pipeline frees Gemma before loading the transformer.

    Returns:
        Estimated peak bytes, or ``None`` when the layout is not LTX MLX or weights are unreadable.
    """
    transformer = ltx_mlx_transformer_file(model_dir)
    if not is_ltx_mlx_layout(model_dir) or transformer is None:
        return None
    try:
        gemma = _files_bfloat16_bytes(model_weight_files(text_encoder_dir))
        connector = _files_bfloat16_bytes([model_dir / _LTX_MLX_CONNECTOR_FILE])
        denoise = _files_bfloat16_bytes([transformer, *(model_dir / name for name in _LTX_MLX_DECODER_FILES)])
    except OSError, ValueError, KeyError, TypeError:
        return None
    peak = max(gemma, denoise) + connector if low_memory else gemma + connector + denoise
    return peak + _VIDEO_WORKING_BYTES


def classify_memory_fit(required_bytes: int, budget_bytes: int) -> str:
    """Classify an estimate against the memory budget as ``fits``, ``tight``, or ``too_large``."""
    if required_bytes <= budget_bytes * _TIGHT_RATIO:
        return "fits"
    if required_bytes <= budget_bytes:
        return "tight"
    return "too_large"


def _image_components(model_dir: Path) -> dict[str, WeightTotals] | None:
    components: dict[str, WeightTotals] = {}
    for path in model_weight_files(model_dir):
        component = component_of(model_dir, path)
        try:
            components[component] = components.get(component, WeightTotals()) + read_safetensors_totals(path)
        except OSError, ValueError, KeyError, TypeError:
            return None
    return components or None


def _image_component_bytes(name: str, totals: WeightTotals, quantize: int | None) -> float:
    if name == "vae":
        return (totals.float_matrix_elements + totals.float_other_elements) * 4 + totals.packed_bytes
    if quantize and not totals.prequantized:
        return totals.float_matrix_elements * (quantize / 8 + _QUANT_GROUP_OVERHEAD_BYTES) + totals.float_other_elements * 2 + totals.packed_bytes
    return totals.bfloat16_bytes


def _files_bfloat16_bytes(paths: list[Path] | tuple[Path, ...]) -> int:
    """Return bytes resident when *paths* load with floats as bfloat16 and packed weights as stored."""
    return sum((read_safetensors_totals(path) for path in paths if path.is_file()), WeightTotals()).bfloat16_bytes
