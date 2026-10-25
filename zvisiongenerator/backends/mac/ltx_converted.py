"""Converted LTX-2.3 folders on macOS: write an MLX folder from a single-file checkpoint.

A converted folder has the layout of dgrauet's MLX folders (``dgrauet/ltx-2.3-mlx-q4`` / ``-q8``), which the vendored
``ltx_pipelines_mlx`` loads:

- ``transformer-distilled.safetensors``: the checkpoint's transformer under the MLX names, with every Linear weight
  inside its blocks quantized by ``mx.quantize`` (affine, groups of 64) and the rest kept as stock stores it
  (``*scale_shift_table*`` in float32, everything else in bfloat16). It is the folder's only ``transformer*`` file,
  so the loader can't pick another one.
- ``connector.safetensors``: linked from the MLX folder when the checkpoint's text connectors equal stock's or it has
  none; otherwise written from the checkpoint (the names map one to one, under ``connector.``).
- ``embedded_config.json``: the checkpoint's own ``config`` metadata in stock's spelling, or stock's when it has none.
- the VAEs, vocoder, upsamplers and their configs, linked from the MLX folder of the same level.
- the folder manifest (:mod:`zvisiongenerator.utils.ltx_converted`), written last.

The transformer is written one tensor at a time into a single file whose header is written first, so the
conversion never holds more than one tensor of it.
"""

from __future__ import annotations

import fnmatch
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.converters.ltx_checkpoint import CHECKPOINT_PREFIX, checkpoint_config, connector_tensors, connectors_status, file_order, transformer_tensors
from zvisiongenerator.utils.file_links import link_or_copy
from zvisiongenerator.utils.ltx_converted import write_converted_ltx
from zvisiongenerator.utils.safetensors_stream import SingleFileWriter, TensorFile

if TYPE_CHECKING:
    import numpy as np
    import torch

    from zvisiongenerator.converters.ltx_checkpoint import LtxCheckpoint

__all__ = ["convert", "write_converted"]

TRANSFORMER_FILE = "transformer-distilled.safetensors"
_CONNECTOR_FILE = "connector.safetensors"
_EMBEDDED_CONFIG = "embedded_config.json"
# What the converted folder links from the MLX folder of its level, by LTX version. Never a transformer file.
_BASE_FILES = {
    "2.3": (
        "audio_vae.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "vocoder.safetensors",
        "config.json",
        "quantize_config.json",
        "split_model.json",
        "spatial_upscaler_*",
        "temporal_upscaler_*",
    ),
    # LTX-2.5 packs bundle the Gemma 4 text encoder and its tokenizer files, two VAE pairs and the duration head.
    # quantize_config.json only marks which parts are quantized; the bits are read from the weights' shapes.
    "2.5": (
        "audio_vae.safetensors",
        "vae_decoder_*.safetensors",
        "vae_encoder_*.safetensors",
        "vocoder.safetensors",
        "duration_head.safetensors",
        "text_encoder.safetensors",
        "text_encoder_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "quantize_config.json",
        "split_model.json",
        "spatial_upscaler_*",
        "temporal_upscaler_*",
    ),
}
_GROUP_SIZE = 64
_STORAGE = "mlx-affine-g64"
_MLX_PREFIX = "transformer."
_CONNECTOR_PREFIX = "connector."
_BLOCKS_PREFIX = f"{CHECKPOINT_PREFIX}transformer_blocks."
# Native → MLX names, as the vendored LTXV_LORA_COMFY_RENAMING_MAP renames them.
_RENAMES = (
    (".to_out.0.", ".to_out."),
    ("audio_ff.net.0.proj.", "audio_ff.proj_in."),
    ("audio_ff.net.2.", "audio_ff.proj_out."),
    (".ff.net.0.proj.", ".ff.proj_in."),
    (".ff.net.2.", ".ff.proj_out."),
    (".linear_1.", ".linear1."),
    (".linear_2.", ".linear2."),
)

Report = Callable[[str, int, int], None]
# Quantizes a weight at a bit width into MLX's ``(packed weight, scales, biases)``, as NumPy arrays (uint32, then bfloat16 bits as uint16).
Quantize = Callable[["torch.Tensor", int], tuple["np.ndarray", "np.ndarray", "np.ndarray"]]


def convert(checkpoint: LtxCheckpoint, target: Path, bits: int, *, config: dict[str, Any], version: str | None, models_dir: Path, lora: Path | None, report: Report) -> None:
    """Write the MLX folder of *checkpoint* at *bits* into *target*, with the other components of LTX version *version*'s MLX folder.

    Connectors that differ from stock's, or are quantized and can't be compared, are written from the checkpoint (with
    stock's for any group it lacks); stock ones are linked from the MLX folder.

    *models_dir* is unused: an MLX folder shares no component with other models' folders.

    Raises:
        RuntimeError: When the MLX folder can't be downloaded.
    """
    import torch

    from zvisiongenerator.backends.mac.memory import mlx_cache_limit
    from zvisiongenerator.utils.config import ltx_mlx_base_repo

    del models_dir
    report("connectors", 0, 1)
    # Connectors that differ from stock's, or are quantized and can't be compared, are kept: a fine-tune may have trained them.
    own_connectors = connectors_status(checkpoint) in ("changed", "unknown")
    report("download", 0, 1)
    base_dir = _base_dir(ltx_mlx_base_repo(config, bits, version), checkpoint.version)
    merge_device = torch.device("mps") if torch.backends.mps.is_available() else None
    # No MLX buffer cache: each tensor is quantized once, and cached buffers would add up to the whole transformer.
    with mlx_cache_limit(0):
        write_converted(checkpoint, target, bits, base_dir=base_dir, own_connectors=own_connectors, lora=lora, quantize=_mlx_quantize, merge_device=merge_device, report=report)


def write_converted(
    checkpoint: LtxCheckpoint,
    target: Path,
    bits: int,
    *,
    base_dir: Path,
    own_connectors: bool,
    lora: Path | None,
    quantize: Quantize,
    merge_device: Any,
    report: Report,
) -> None:
    """Write the MLX folder of *checkpoint* at *bits* into the new folder *target*.

    Args:
        checkpoint: The checked checkpoint.
        target: The folder to write (a partial folder; the caller moves it into place).
        bits: 4 or 8.
        base_dir: The MLX folder of that level the other components come from.
        own_connectors: Write the checkpoint's own text connectors, with stock's for any group it lacks, instead of linking stock's.
        lora: Lightricks' distilled LoRA to merge in, or ``None``.
        quantize: Quantizes one block weight (``mx.quantize`` at groups of 64).
        merge_device: Where the LoRA merge is computed, or ``None`` for the CPU.
        report: Called with ``(stage, done, total)`` as the conversion advances.

    Raises:
        ValueError: When a block weight can't be split into groups of 64, or the connectors have tensors stock's lack or another shape.
    """
    if own_connectors:
        _write_connectors(checkpoint, target / _CONNECTOR_FILE, base_dir / _CONNECTOR_FILE)
    else:
        link_or_copy(base_dir / _CONNECTOR_FILE, target / _CONNECTOR_FILE)
    _write_transformer(checkpoint, target / TRANSFORMER_FILE, bits, lora=lora, quantize=quantize, merge_device=merge_device, report=report)
    report("components", 0, 1)
    stock_config = json.loads((base_dir / _EMBEDDED_CONFIG).read_text(encoding="utf-8"))
    (target / _EMBEDDED_CONFIG).write_text(json.dumps(embedded_config(checkpoint_config(checkpoint), stock_config), indent=2) + "\n", encoding="utf-8")
    _link_components(base_dir, target, checkpoint.version)
    block_matrices, transformer_rest = _transformer_elements(checkpoint)
    write_converted_ltx(
        target,
        {
            "bits": bits,
            "storage": _STORAGE,
            "block_matrices": block_matrices,
            "transformer_rest": transformer_rest,
            "source": checkpoint.path.name,
            "distilled_lora": lora.name if lora is not None else None,
        },
        version=checkpoint.version,
    )


def embedded_config(checkpoint: dict[str, Any] | None, stock: dict[str, Any]) -> dict[str, Any]:
    """Return the ``embedded_config.json`` of a converted folder: the checkpoint's config in stock's spelling, or *stock* when it has none.

    A string value that differs from stock's only in case (e.g. ``PER_TOKEN_RMS`` for ``per_token_rms``) takes
    stock's spelling, and stock's sections and keys the checkpoint lacks (such as LTX-2.5's ``model_version``) are kept.
    """
    if checkpoint is None:
        return stock
    return {**stock, **{section: _stock_spelling(values, stock.get(section)) for section, values in checkpoint.items()}}


def _stock_spelling(values: Any, stock: Any) -> Any:
    """Return the config section *values* over stock's section, each string that equals stock's but for case spelled as in *stock*."""
    if not isinstance(values, dict) or not isinstance(stock, dict):
        return values
    return stock | {
        key: stock[key] if isinstance(value, str) and isinstance(stock.get(key), str) and value != stock[key] and value.lower() == stock[key].lower() else value for key, value in values.items()
    }


def mlx_name(normalized: str) -> str:
    """Return the MLX transformer name of the normalized transformer name *normalized*."""
    name = normalized.removeprefix(CHECKPOINT_PREFIX)
    for old, new in _RENAMES:
        name = name.replace(old, new)
    return _MLX_PREFIX + name


def _quantized(normalized: str, shape: list[int]) -> bool:
    """Return whether the transformer tensor *normalized* is quantized: every Linear weight inside the blocks."""
    return normalized.startswith(_BLOCKS_PREFIX) and normalized.endswith(".weight") and len(shape) == 2


def _stored_dtype(normalized: str) -> str:
    """Return the dtype an unquantized transformer tensor is stored in: float32 for the modulation tables, as stock stores them, else bfloat16."""
    return "F32" if "scale_shift_table" in normalized.rsplit(".", 1)[-1] else "BF16"


def transformer_entries(checkpoint: LtxCheckpoint, order: Iterable[str], bits: int) -> list[tuple[str, str, list[int]]]:
    """Return the converted transformer file's tensors ``(name, dtype, shape)`` for the transformer tensors in *order*.

    Raises:
        ValueError: When a block weight's input size is not a multiple of the group size.
    """
    entries: list[tuple[str, str, list[int]]] = []
    for normalized in order:
        shape = checkpoint.shapes[normalized]
        name = mlx_name(normalized)
        if not _quantized(normalized, shape):
            entries.append((name, _stored_dtype(normalized), shape))
            continue
        rows, columns = shape
        if columns % _GROUP_SIZE:
            raise ValueError(f"{normalized} has {columns} inputs, which can't be quantized in groups of {_GROUP_SIZE}.")
        base = name.removesuffix(".weight")
        groups = [rows, columns // _GROUP_SIZE]
        entries += [(f"{base}.weight", "U32", [rows, columns * bits // 32]), (f"{base}.scales", "BF16", groups), (f"{base}.biases", "BF16", groups)]
    return entries


def _write_transformer(checkpoint: LtxCheckpoint, path: Path, bits: int, *, lora: Path | None, quantize: Quantize, merge_device: Any, report: Report) -> None:
    """Write the converted transformer file of *checkpoint* at *bits* to *path*, one tensor at a time."""
    order = file_order(checkpoint, checkpoint.transformer)
    tensors = transformer_tensors(checkpoint, lora=lora, merge_device=merge_device, progress=lambda done, total: report("transformer", done, total))
    with SingleFileWriter(path, transformer_entries(checkpoint, order, bits)) as writer:
        for normalized, tensor in tensors:
            name = mlx_name(normalized)
            if not _quantized(normalized, checkpoint.shapes[normalized]):
                writer.write(name, _tensor_bytes(tensor, _stored_dtype(normalized)))
                continue
            base = name.removesuffix(".weight")
            for suffix, part in zip((".weight", ".scales", ".biases"), quantize(tensor, bits), strict=True):
                writer.write(base + suffix, part)


def _write_connectors(checkpoint: LtxCheckpoint, path: Path, stock: Path) -> None:
    """Write *checkpoint*'s text connectors to *path* in the MLX connector layout (bfloat16), with stock's (the file *stock*) for every tensor it lacks.

    Transformer-only exports often leave out a group, such as ``text_embedding_projection``; stock's stands in for it.

    Raises:
        ValueError: When the checkpoint has a connector tensor stock's lacks, or one of another shape.
    """
    with TensorFile(stock) as stock_file:
        own = [_connector_name(name) for name in file_order(checkpoint, checkpoint.connectors)]
        shapes = {_connector_name(name): checkpoint.shapes[name] for name in checkpoint.connectors}
        odd = [name for name in own if stock_file.header.get(name, {}).get("shape") != shapes[name]]
        if odd:
            raise ValueError(f"{checkpoint.path.name}'s text connector {odd[0]} is not in stock's, or has another shape; it can't be converted.")
        gaps = sorted(set(stock_file.header) - set(own))
        entries = [(name, "BF16", shapes[name]) for name in own] + [(name, "BF16", stock_file.header[name]["shape"]) for name in gaps]
        with SingleFileWriter(path, entries) as writer:
            for normalized, tensor in connector_tensors(checkpoint):
                writer.write(_connector_name(normalized), _tensor_bytes(tensor, "BF16"))
            for name in gaps:
                writer.write(name, _tensor_bytes(stock_file.read(name), "BF16"))


def _connector_name(normalized: str) -> str:
    """Return the ``connector.safetensors`` name of the normalized connector name *normalized*."""
    return _CONNECTOR_PREFIX + normalized.removeprefix(CHECKPOINT_PREFIX)


def _tensor_bytes(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    """Return *tensor*'s bytes in the safetensors dtype *dtype* (``BF16`` or ``F32``)."""
    import torch

    stored = tensor.to(torch.float32 if dtype == "F32" else torch.bfloat16).contiguous().reshape(-1)
    return stored.view(torch.uint8).numpy()


def _mlx_quantize(weight: torch.Tensor, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize *weight* in bfloat16 with ``mx.quantize`` at groups of 64; return the packed weight, scales and biases as NumPy arrays."""
    import mlx.core as mx
    import numpy as np
    import torch

    source = mx.array(weight.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()).view(mx.bfloat16)
    packed, scales, biases = mx.quantize(source, group_size=_GROUP_SIZE, bits=bits)
    return np.array(packed), np.array(scales.astype(mx.bfloat16).view(mx.uint16)), np.array(biases.astype(mx.bfloat16).view(mx.uint16))


def _transformer_elements(checkpoint: LtxCheckpoint) -> tuple[int, int]:
    """Count the elements of *checkpoint*'s transformer: ``(Linear weights in its blocks, the rest)``."""
    blocks = rest = 0
    for normalized, shape in ((name, checkpoint.shapes[name]) for name in checkpoint.transformer):
        count = math.prod(shape)
        if _quantized(normalized, shape):
            blocks += count
        else:
            rest += count
    return blocks, rest


def _link_components(base_dir: Path, target: Path, version: str) -> None:
    """Hard-link the LTX-*version* MLX folder's components in ``_BASE_FILES`` into *target*, copying where a link is impossible."""
    for source in sorted(base_dir.iterdir()):
        if source.is_file() and any(fnmatch.fnmatchcase(source.name, pattern) for pattern in _BASE_FILES[version]):
            link_or_copy(source, target / source.name)


def _base_dir(reference: str, version: str) -> Path:
    """Return the local folder of the LTX-*version* MLX repo *reference* (``repo`` or ``repo@revision``) with the files a converted folder takes from it, downloading what is missing.

    Raises:
        RuntimeError: When they can't be downloaded.
    """
    from huggingface_hub import snapshot_download

    from zvisiongenerator.utils.paths import split_huggingface_revision

    repo, revision = split_huggingface_revision(reference)
    patterns = [*_BASE_FILES[version], _EMBEDDED_CONFIG, _CONNECTOR_FILE]
    try:
        return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
    except Exception as exc:  # noqa: BLE001 - any download failure stops the conversion with one message
        raise RuntimeError(f"Converting needs the VAEs, vocoder and configs of {repo}, which couldn't be downloaded: {exc}") from exc
