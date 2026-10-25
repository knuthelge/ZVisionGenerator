"""Decode the quantized weights of community checkpoints into plain floating tensors, one weight at a time.

ComfyUI-style checkpoints store a quantized linear weight ``<module>.weight`` next to small companion tensors named
``<module>.<part>``: a ``comfy_quant`` record (JSON bytes) naming the format, and the scales or codebook that format
needs. The formats read here:

- ``float8_e4m3fn`` / ``float8_e5m2``: FP8 times ``weight_scale`` (one value, or one per output row). Older files
  name the scale ``scale_weight`` and carry no record.
- ``int8_tensorwise``: INT8 times ``weight_scale`` (one value, or one per output row).
- ``convrot_w4a4``: signed 4-bit codes, two per byte, times a per-row ``weight_scale``.
- ``asym_w4a8_int8`` / ``w6a8_int8``: 4-bit codebook (or 6-bit uniform) codes, scaled per group of 16 by the FP8
  ``weight_s_rel`` onto the INT8 grid, then per row by ``weight_s_channel``.
- ``nvfp4``: FP4 (E2M1) values, two per byte with the first in the high nibble, times FP8 block scales of 16
  (``weight_scale``, in cuBLAS's tiled layout) and a per-tensor ``weight_scale_2``.
- ``mxfp8``: FP8 values times power-of-two (E8M0) block scales of 32 (``weight_scale``, in cuBLAS's tiled layout).

A ``pre_quant_scale`` (an activation smoothing some NVFP4 files carry) is multiplied into the weight's columns.

ConvRot weights (``int8_tensorwise`` with ``convrot``, and every 4- and 6-bit format) were multiplied by a block
Hadamard matrix over the input features before they were quantized. The rotation is symmetric and its own inverse,
so applying it again returns the weight to its own basis. Decoding follows comfy-kitchen's portable reference, so a
weight decodes to the values ComfyUI computes with.

Other formats are refused by name. Every decoded weight is bfloat16; tensors that are not quantized pass through
unchanged, except an FP8 tensor with no scale, which is cast to bfloat16.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.safetensors_stream import TensorFile, iter_tensors, read_header

if TYPE_CHECKING:
    import torch

__all__ = ["AUX_SUFFIXES", "DECODED_FORMATS", "decode_weight", "decoded_shape", "decoded_tensors", "precision_label", "quantization_formats"]

# How a checkpoint inspection names each quantization format.
_FORMAT_LABELS = {
    "float8_e4m3fn": "FP8",
    "float8_e5m2": "FP8 (E5M2)",
    "int8_tensorwise": "INT8",
    "convrot_w4a4": "INT4 (W4A4)",
    "asym_w4a8_int8": "INT4 (W4A8)",
    "w6a8_int8": "INT6 (W6A8)",
    "nvfp4": "NVFP4",
    "mxfp8": "MXFP8",
}

# Companion tensors of a quantized weight: never yielded on their own.
AUX_SUFFIXES = (
    ".weight_scale",
    ".scale_weight",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
    ".comfy_quant",
    ".weight_codebook",
    ".weight_s_channel",
    ".weight_s_rel",
)
_SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
_FP8_FORMATS = ("float8_e4m3fn", "float8_e5m2")
_GROUPED_FORMATS = ("asym_w4a8_int8", "w6a8_int8")
DECODED_FORMATS = (*_FP8_FORMATS, "int8_tensorwise", "convrot_w4a4", *_GROUPED_FORMATS, "nvfp4", "mxfp8")
# FP4 (E2M1) values by code.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# What an unlabelled scaled weight is read as, by the dtype it is stored in.
_UNLABELLED = {"F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
# The companion tensors each format stores as FP8 bits, sometimes under a uint8 dtype.
_FP8_PARTS: dict[Any, tuple[str, ...]] = {"asym_w4a8_int8": (".weight_s_rel",), "w6a8_int8": (".weight_s_rel",), "nvfp4": (".weight_scale",)}
_DEFAULT_CONVROT_GROUP = 256
_DEFAULT_GROUP = 16


def _module(name: str) -> str:
    """Return the module part of a tensor name (everything before its last dot)."""
    return name.rsplit(".", 1)[0]


def precision_label(precision: dict[str, int]) -> str:
    """Return how a checkpoint's transformer weights are stored, from a format → weight count tally (empty when none is quantized): ``bf16``, or its formats with counts."""
    if not precision:
        return "bf16"
    return ", ".join(f"{_FORMAT_LABELS.get(name, name)} ({count})" for name, count in sorted(precision.items(), key=lambda item: -item[1]))


def quantization_formats(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Return each quantized weight's name → its quantization record, read from the headers and the ``comfy_quant`` tensors.

    A weight with a ``comfy_quant`` record gets that record; a scaled FP8 weight without one (older files) gets
    ``{"format": "float8_e4m3fn"}`` or ``float8_e5m2``. Unquantized weights are left out.

    Raises:
        ValueError: When a record names a format this module can't decode (naming it), when an INT8 weight has no
            record (its scale alone doesn't say whether it was rotated), or a weight's companion tensors are missing
            or don't fit it.
    """
    files = list(paths)
    headers = {path: read_header(path)[1] for path in files}
    every = {name: info for header in headers.values() for name, info in header.items()}
    records: dict[str, dict[str, Any]] = {}
    for path in files:
        wanted = [name for name in headers[path] if name.endswith(".comfy_quant")]
        for name, tensor in iter_tensors(path, set(wanted).__contains__):
            try:
                record = json.loads(bytes(tensor.tolist()).decode())
            except ValueError as exc:
                raise ValueError(f"{name} in {path.name} is not a readable quantization record.") from exc
            records[f"{_module(name)}.weight"] = record if isinstance(record, dict) else {}
    for name in every:
        if name.endswith(_SCALE_SUFFIXES):
            weight = f"{_module(name)}.weight"
            if weight not in records and weight in every:
                dtype = every[weight]["dtype"]
                if dtype not in _UNLABELLED:
                    raise ValueError(f"{weight} is stored as {dtype} with a scale but no quantization record, so how it was quantized can't be told; it can't be read.")
                records[weight] = {"format": _UNLABELLED[dtype]}
    for weight, record in records.items():
        quant_format = record.get("format")
        if quant_format not in DECODED_FORMATS:
            raise ValueError(f"{weight} is quantized as {quant_format!r}, which can't be read (readable: {', '.join(DECODED_FORMATS)}).")
        missing = [part for part in _required_parts(record) if f"{_module(weight)}{part}" not in every]
        if weight not in every or missing:
            raise ValueError(f"{weight} is quantized as {quant_format!r} but its {', '.join(missing) or 'weight'} is missing.")
        _check_shapes(weight, record, every)
    return records


def _check_shapes(weight: str, record: Mapping[str, Any], header: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a quantized *weight* whose companion tensors in *header* don't fit it, before any tensor is read.

    Raises:
        ValueError: Naming the weight, when a scale is neither one value nor one per row, grouped scales don't fit the
            packed codes, or block scales aren't the tiled layout of the weight's blocks.
    """
    quant_format = record.get("format")
    module = _module(weight)
    shape = list(header[weight]["shape"])
    rows = shape[0] if shape else 1

    def _size(suffix: str) -> int:
        return math.prod(header[f"{module}{suffix}"]["shape"])

    if quant_format in _GROUPED_FORMATS:
        params = record.get("params") if isinstance(record.get("params"), dict) else {}
        group = int(record.get("group_size", params.get("group_size", _DEFAULT_GROUP)))
        s_rel = list(header[f"{module}.weight_s_rel"]["shape"])
        columns = s_rel[-1] * group if len(s_rel) == 2 else 0
        fits = len(shape) == 2 and s_rel[0] == rows and _size(".weight_s_channel") == rows and any(columns * bits == shape[1] * 8 for bits in (4, 6))
        if not fits:
            raise ValueError(f"{weight}'s packed weight {shape} does not fit its group scales {s_rel} at 4 or 6 bits.")
    elif quant_format in ("nvfp4", "mxfp8"):
        columns, block = (shape[1] * 2, 16) if quant_format == "nvfp4" else (shape[1], 32)
        tiled = -(-rows // 128) * 128 * -(-(columns // block) // 4) * 4
        if len(shape) != 2 or columns % block or _size(".weight_scale") != tiled:
            raise ValueError(f"The block scales of {weight} hold {_size('.weight_scale')} values, not the {tiled} of its {rows}x{columns // block} blocks in the tiled layout.")
    else:
        scale = next((f"{module}{suffix}" for suffix in _SCALE_SUFFIXES if f"{module}{suffix}" in header), None)
        if scale is not None and _size(scale.removeprefix(module)) not in (1, rows):
            raise ValueError(f"The scale of {weight} has shape {list(header[scale]['shape'])} for a weight of shape {shape}; only one scale per tensor or per row can be read.")


def decoded_shape(header: Mapping[str, Mapping[str, Any]], name: str, record: Mapping[str, Any] | None) -> list[int]:
    """Return the shape weight *name* of *header* has once decoded: its stored shape, unless 4- or 6-bit codes pack several to a byte.

    Args:
        header: The file's header (or the merged headers of its shards).
        name: The weight's name.
        record: Its quantization record, ``None`` when it isn't quantized.
    """
    shape = list(header[name]["shape"])
    quant_format = (record or {}).get("format")
    if quant_format in ("convrot_w4a4", "nvfp4"):
        return [shape[0], shape[1] * 2]
    if quant_format in _GROUPED_FORMATS:
        params = record.get("params") if isinstance(record.get("params"), dict) else {}
        group = int(record.get("group_size", params.get("group_size", _DEFAULT_GROUP)))
        return [shape[0], header[f"{_module(name)}.weight_s_rel"]["shape"][1] * group]
    return shape


def _required_parts(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the companion tensors a weight of *record*'s format can't be decoded without."""
    quant_format = record.get("format")
    if quant_format in _GROUPED_FORMATS:
        return (".weight_s_rel", ".weight_s_channel")
    if quant_format in _FP8_FORMATS:
        return ()  # the scale may be named either way, or be absent (a plain cast)
    if quant_format == "nvfp4":
        return (".weight_scale", ".weight_scale_2")
    return (".weight_scale",)


def decoded_tensors(
    paths: Iterable[Path],
    wanted: Callable[[str], bool] | None = None,
    *,
    device: Any = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` of the safetensors files *paths* with every quantized weight decoded, in file order.

    Companion tensors (scales, codebooks, records) are left out. A weight's companions are read by name as it comes,
    and may sit in another file of *paths* (another shard).

    Args:
        paths: The files of one checkpoint.
        wanted: Accepts the names to yield (a weight's, not a companion's); every name when ``None``.
        device: Where the decoding runs (a GPU is much faster for rotated weights); the CPU by default. Decoded
            weights are returned on the CPU.

    Raises:
        ValueError: As :func:`quantization_formats` does.
    """
    files = list(paths)
    records = quantization_formats(files)
    owners = {name: path for path in files for name in read_header(path)[1]}
    readers: dict[Path, TensorFile] = {}

    def _part(name: str) -> torch.Tensor | None:
        path = owners.get(name)
        if path is None:
            return None
        if path not in readers:
            readers[path] = TensorFile(path)
        return readers[path].read(name)

    def _accepted(name: str) -> bool:
        return not name.endswith(AUX_SUFFIXES) and (wanted is None or wanted(name))

    try:
        for path in files:
            for name, tensor in iter_tensors(path, _accepted):
                record = records.get(name)
                if record is None:
                    yield name, _plain(tensor)
                    continue
                module = _module(name)
                parts = {suffix: part for suffix in AUX_SUFFIXES if suffix != ".comfy_quant" and (part := _part(f"{module}{suffix}")) is not None}
                yield name, decode_weight(name, tensor, record, parts, device=device)
    finally:
        for reader in readers.values():
            reader.close()


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    """Return *tensor*, cast to bfloat16 when it is FP8 (which nothing downstream computes in)."""
    import torch

    return tensor.to(torch.bfloat16) if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else tensor


def decode_weight(name: str, weight: torch.Tensor, record: Mapping[str, Any], parts: Mapping[str, torch.Tensor], *, device: Any = None) -> torch.Tensor:
    """Return the quantized *weight* decoded to bfloat16 in its own basis, on the CPU.

    Args:
        name: The weight's name, for error messages.
        weight: The stored weight.
        record: Its quantization record (``format`` plus the format's settings).
        parts: Its companion tensors by suffix (``.weight_scale``, ``.weight_s_rel`` ...).
        device: Where to decode; the CPU by default.

    Raises:
        ValueError: When the format is unknown or the tensors don't fit it.
    """
    import torch

    quant_format = record.get("format")
    params = record.get("params") if isinstance(record.get("params"), dict) else {}

    def _setting(key: str, default: Any) -> Any:
        return record.get(key, params.get(key, default))

    weight = _onto(weight, device)
    on = {suffix: _onto(part.view(torch.float8_e4m3fn) if suffix in _FP8_PARTS.get(quant_format, ()) and part.dtype == torch.uint8 else part, device) for suffix, part in parts.items()}
    if quant_format in _FP8_FORMATS:
        scale = on.get(".weight_scale", on.get(".scale_weight"))
        decoded = weight.to(torch.float32) if scale is None else weight.to(torch.float32) * _row_scale(name, weight, scale)
    elif quant_format == "int8_tensorwise":
        decoded = weight.to(torch.float32) * _row_scale(name, weight, on[".weight_scale"])
        if _setting("convrot", False):
            decoded = _unrotated(name, decoded, int(_setting("convrot_groupsize", _DEFAULT_CONVROT_GROUP)))
    elif quant_format == "convrot_w4a4":
        codes = _signed_nibbles(weight)
        decoded = codes.to(torch.float32) * _row_scale(name, codes, on[".weight_scale"])
        decoded = _unrotated(name, decoded, int(_setting("convrot_groupsize", _DEFAULT_CONVROT_GROUP)))
    elif quant_format in _GROUPED_FORMATS:
        decoded = _grouped(name, weight, on, int(_setting("group_size", _DEFAULT_GROUP)))
        decoded = _unrotated(name, decoded, int(_setting("convrot_groupsize", _DEFAULT_CONVROT_GROUP)))
    elif quant_format == "nvfp4":
        decoded = _nvfp4(name, weight, on[".weight_scale"], on[".weight_scale_2"])
    elif quant_format == "mxfp8":
        decoded = _mxfp8(name, weight, on[".weight_scale"])
    else:
        raise ValueError(f"{name} is quantized as {quant_format!r}, which can't be read.")
    if ".pre_quant_scale" in on:
        decoded = decoded * on[".pre_quant_scale"].to(torch.float32).reshape(1, -1)
    return decoded.to(torch.bfloat16).cpu()


def _onto(tensor: torch.Tensor, device: Any) -> torch.Tensor:
    """Return *tensor* on *device* (where it is, for ``None``), FP8 widened to float32 and E8M0 exponents as bytes first: not every device holds FP8 (Apple's MPS does not)."""
    import torch

    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == getattr(torch, "float8_e8m0fnu", None):
        tensor = tensor.view(torch.uint8)  # power-of-two exponents, read as bits
    return tensor if device is None else tensor.to(device)


def _row_scale(name: str, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return *scale* shaped to multiply *weight*: a single value, or one value per output row as a column."""
    import torch

    scale = scale.to(torch.float32)
    if scale.numel() == 1:
        return scale.reshape(())
    if weight.ndim == 2 and scale.numel() == weight.shape[0]:
        return scale.reshape(-1, 1)
    raise ValueError(f"The scale of {name} has shape {list(scale.shape)} for a weight of shape {list(weight.shape)}; only one scale per tensor or per row can be read.")


def _signed_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Return the signed 4-bit codes of *packed* (two per byte, the low nibble first) as int8 ``[rows, 2 * columns]``."""
    import torch

    wide = packed.view(torch.uint8).to(torch.int16)
    codes = torch.stack([wide & 0x0F, (wide >> 4) & 0x0F], dim=-1)
    codes = torch.where(codes >= 8, codes - 16, codes)
    return codes.reshape(packed.shape[0], -1).to(torch.int8)


def _grouped(name: str, packed: torch.Tensor, parts: Mapping[str, torch.Tensor], group_size: int) -> torch.Tensor:
    """Decode a W4A8 / W6A8 weight to its rotated floating values: codes → INT8 grid (per group) → per-row scale."""
    import torch

    s_rel = parts[".weight_s_rel"]
    rows, groups = s_rel.shape
    columns = groups * group_size
    bits = packed.shape[1] * 8 // columns if columns else 0
    if packed.ndim != 2 or packed.shape[0] != rows or bits not in (4, 6) or columns * bits != packed.shape[1] * 8:
        raise ValueError(f"{name}'s packed weight {list(packed.shape)} does not fit its group scales {list(s_rel.shape)} at 4 or 6 bits.")
    codes = _unsigned_codes(packed, columns, bits)
    codebook = parts.get(".weight_codebook")
    if codebook is not None:
        values = codebook.to(torch.float32)[codes]
    else:
        values = codes.to(torch.float32) - float(1 << (bits - 1))
    grid = (values.view(rows, groups, group_size) * s_rel.to(torch.float32).unsqueeze(-1)).round_().clamp_(-127, 127)
    return grid.view(rows, columns) * parts[".weight_s_channel"].to(torch.float32).reshape(-1, 1)


def _unsigned_codes(packed: torch.Tensor, columns: int, bits: int) -> torch.Tensor:
    """Return the unsigned codes of a W4A8 / W6A8 row layout: a nibble plane (even column = low nibble), then at 6 bits a plane of each code's top two bits."""
    import torch

    wide = packed.view(torch.uint8).to(torch.int64)
    low = wide[:, : columns // 2]
    codes = torch.empty(packed.shape[0], columns, dtype=torch.int64, device=packed.device)
    codes[:, 0::2] = low & 0x0F
    codes[:, 1::2] = (low >> 4) & 0x0F
    if bits == 6:
        shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.int64)
        high = (wide[:, columns // 2 :].unsqueeze(-1) >> shifts) & 0x3
        codes |= high.reshape(packed.shape[0], columns) << 4
    return codes


def _nvfp4(name: str, packed: torch.Tensor, block_scales: torch.Tensor, tensor_scale: torch.Tensor) -> torch.Tensor:
    """Decode an NVFP4 weight: E2M1 codes (high nibble first) times the unswizzled FP8 scale of each block of 16, times the per-tensor scale."""
    import torch

    wide = packed.view(torch.uint8).to(torch.int64)
    codes = torch.stack([wide >> 4, wide & 0x0F], dim=-1).reshape(packed.shape[0], -1)
    values = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)[codes]
    rows, columns = values.shape
    scales = _from_blocked(name, block_scales.to(torch.float32), rows, columns // 16)
    return (values.view(rows, columns // 16, 16) * scales.unsqueeze(-1)).view(rows, columns) * tensor_scale.to(torch.float32).reshape(())


def _mxfp8(name: str, weight: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """Decode an MXFP8 weight: FP8 values times the unswizzled power-of-two (E8M0) scale of each block of 32."""
    import torch

    rows, columns = weight.shape
    exponents = _from_blocked(name, block_scales.view(torch.uint8).to(torch.int32), rows, columns // 32)
    scales = torch.where(exponents == 0, torch.zeros_like(exponents), exponents << 23).view(torch.float32)
    return (weight.to(torch.float32).view(rows, columns // 32, 32) * scales.unsqueeze(-1)).view(rows, columns)


def _from_blocked(name: str, blocked: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Return block scales stored in cuBLAS's tiled layout (128-row by 4-column tiles, padded) as a plain ``[rows, columns]`` matrix."""
    row_tiles, column_tiles = -(-rows // 128), -(-columns // 4)
    if blocked.numel() != row_tiles * 128 * column_tiles * 4:
        raise ValueError(f"The block scales of {name} hold {blocked.numel()} values, not the {row_tiles * 128 * column_tiles * 4} of a {rows}x{columns} tiled layout.")
    tiles = blocked.reshape(-1, 32, 4, 4).transpose(1, 2).reshape(row_tiles, column_tiles, 128, 4)
    return tiles.permute(0, 2, 1, 3).reshape(row_tiles * 128, column_tiles * 4)[:rows, :columns]


def _unrotated(name: str, weight: torch.Tensor, group: int) -> torch.Tensor:
    """Return the ConvRot-rotated *weight* in its own basis: each group of *group* input features times the (self-inverse) Hadamard matrix."""
    rows, columns = weight.shape
    if columns % group:
        raise ValueError(f"{name} has {columns} input features, which the rotation group of {group} does not divide.")
    hadamard = _hadamard(group).to(device=weight.device)
    return (weight.reshape(rows, columns // group, group) @ hadamard.T).reshape(rows, columns)


@cache
def _hadamard(size: int) -> torch.Tensor:
    """Return the normalized regular Hadamard matrix of *size* (a power of 4) that ConvRot rotates with."""
    import torch

    if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
        raise ValueError(f"A ConvRot rotation group must be a power of 4, not {size}.")
    base = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    matrix = base
    while matrix.shape[0] < size:
        matrix = torch.kron(matrix, base)
    return matrix / math.sqrt(size)
