"""Convert Ideogram 4 Instant's single-file BF16 transformer into the layout a backend loads.

The source file uses the base FP8 repo's tensor names: attention as a fused ``qkv`` (q, k and v stacked by rows)
and ``o``. Two output formats:

- ``bf16``: diffusers' ``Ideogram4Transformer2DModel`` names (``to_q``/``to_k``/``to_v``/``to_out.0``), in BF16.
- ``fp8``: the base FP8 repo's names, each linear weight stored as ``float8_e4m3fn`` with a per-row float32
  ``weight_scale``, the layout mflux's ``Fp8Linear`` reads. Biases, norms and the image-indicator embedding stay BF16.

Tensors are read, converted and written one at a time, so the 18.6 GB model is never held in memory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.safetensors_stream import ShardWriter, iter_tensors

if TYPE_CHECKING:
    import torch

__all__ = ["convert_tensor", "fp8_rows", "transformer_config", "write_transformer"]

_FP8_MAX = 448.0  # largest finite float8_e4m3fn value
_SHARD_PREFIX = "diffusion_pytorch_model"
_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
# The one 2-D weight the base FP8 repo keeps in BF16: an embedding table, not a linear layer.
_BF16_WEIGHTS = frozenset({"embed_image_indicator.weight"})
# Config keys that describe a particular copy's storage, not the architecture.
_COPY_CONFIG_KEYS = ("quantization_config", "_name_or_path")


def fp8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return *weight* as ``float8_e4m3fn`` codes and the per-row float32 scale that restores it (``codes * scale``).

    Each row is scaled so its largest magnitude maps to the largest FP8 value; an all-zero row gets scale 1.
    """
    import torch

    rows = weight.to(torch.float32)
    absmax = rows.abs().amax(dim=1)
    scale = torch.where(absmax > 0, absmax / _FP8_MAX, torch.ones_like(absmax))
    codes = (rows / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def convert_tensor(name: str, tensor: torch.Tensor, transformer_format: str) -> dict[str, torch.Tensor]:
    """Return the tensors that source tensor *name* becomes in *transformer_format* (``bf16`` or ``fp8``).

    Raises:
        ValueError: For an unknown format.
    """
    import torch

    tensor = tensor.to(torch.bfloat16)
    if transformer_format == "bf16":
        if name.endswith(".attention.qkv.weight"):
            prefix = name.removesuffix("qkv.weight")
            query, key, value = tensor.chunk(3, dim=0)
            return {prefix + "to_q.weight": query, prefix + "to_k.weight": key, prefix + "to_v.weight": value}
        if name.endswith(".attention.o.weight"):
            return {name.removesuffix("o.weight") + "to_out.0.weight": tensor}
        return {name: tensor}
    if transformer_format == "fp8":
        if tensor.ndim == 2 and name.endswith(".weight") and name not in _BF16_WEIGHTS:
            codes, scale = fp8_rows(tensor)
            return {name: codes, name.removesuffix("weight") + "weight_scale": scale}
        return {name: tensor}
    raise ValueError(f"Unknown Ideogram 4 transformer format {transformer_format!r}; expected 'bf16' or 'fp8'.")


def transformer_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """Return the base model's transformer config without the keys that describe its own storage (its quantization)."""
    return {key: value for key, value in base_config.items() if key not in _COPY_CONFIG_KEYS}


def write_transformer(
    source_file: Path,
    target_dir: Path,
    transformer_format: str,
    base_config: dict[str, Any],
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Write the transformer in *source_file* into *target_dir* as sharded safetensors with its config.

    Args:
        source_file: The single-file BF16 transformer.
        target_dir: The ``transformer`` folder to create.
        transformer_format: ``bf16`` or ``fp8`` (see :func:`convert_tensor`).
        base_config: The base model's ``transformer/config.json``; the architecture is the same.
        cancelled: Checked before each tensor; when it turns true the write stops.

    Returns:
        Whether the write finished (``False`` when *cancelled* stopped it, leaving a partial folder).
    """
    writer = ShardWriter(target_dir, _SHARD_PREFIX, _INDEX_NAME)
    for name, tensor in iter_tensors(source_file):
        if cancelled is not None and cancelled():
            return False
        writer.add(convert_tensor(name, tensor, transformer_format))
    writer.close()
    (target_dir / "config.json").write_text(json.dumps(transformer_config(base_config), indent=2) + "\n", encoding="utf-8")
    return True
