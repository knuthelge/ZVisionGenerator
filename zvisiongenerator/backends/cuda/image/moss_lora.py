"""Unmerged LoRA and LoKr adapters for the MOSS-Image transformer.

Adapters are added to a layer's output (``W x + scale * up(down(x))``) instead of being merged into its weights, so
they work the same on bfloat16 and FP8-stored layers and never change them. The layer wrappers and the LoKr term are
shared with the diffusers backend (:mod:`.lokr`).

Supported files are the formats seen in MOSS-Image LoRAs in the wild:

* kohya, dotted (``diffusion_model.layers.N.proj.q.lora_down.weight``) or underscore (``lora_unet_layers_N_proj_q``),
  with ``alpha`` tensors;
* ``lora_A``/``lora_B`` under ``diffusion_model.`` or ``model.diffusion_model.`` (no alpha), ``transformer.`` (alpha in
  the file's ``lora_adapter_metadata``) or ``base_model.model.`` (PEFT adapter files), with or without a PEFT adapter
  name (``lora_A.default.weight``), which is dropped on reading;
* LoKr (``lokr_w1``/``lokr_w2``);
* a fused ``proj.qkv`` layer in any of these is split over the model's separate q, k and v layers.

LoHa, full-weight ``.diff`` deltas, Tucker and DoRA layers are skipped with a warning.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from zvisiongenerator.backends.cuda.image.lokr import LoKrTerm, LoraParts, adapt_linear, base_linear, check_fits, lokr_factors, split_key, split_lora_tensors
from zvisiongenerator.backends.cuda.lora_peft import warn_unmatched_lora
from zvisiongenerator.utils.lora_files import normalised_lora_state

__all__ = ["LoraFile", "install_loras", "moss_targets"]

# Layer-name prefixes of the supported formats (those utils.lora_family accepts, plus kohya's); at most one is
# stripped, the longest match first so ``model.diffusion_model.`` is not left half-stripped.
_PREFIXES = ("model.diffusion_model.", "base_model.model.", "diffusion_model.", "transformer.", "lora_unet_")
_FAMILY = "moss"
_LOADER = "moss"
_FUSED_SUFFIX = ".proj.qkv"
_TOKEN_CHUNK = 4096
# Tensors of layers that only an unsupported variant of the adapter can use.
_UNSUPPORTED_VARIANT_TENSORS = frozenset({"lokr_t2", "dora_scale"})

# Output rows of one target, taken from a fused layer (None: all rows).
_Target = tuple[str, slice | None]


@dataclass(frozen=True)
class LoraFile:
    """A LoRA file to install: its path, its strength and the dtype its factors compute in."""

    path: str
    strength: float
    dtype: torch.dtype


@dataclass(frozen=True)
class _Adapter:
    """One layer of a file: the shape of its full delta and a factory for the term of a set of its output rows."""

    shape: tuple[int, int]
    make: Callable[[slice | None], torch.nn.Module]


class _LoraTerm(torch.nn.Module):
    """``scale * up(down(x))``, computed in chunks of tokens so large inputs never build a huge intermediate.

    The factors are plain attributes on the GPU: model offloading neither tracks nor moves them.
    """

    def __init__(self, down: torch.Tensor, up: torch.Tensor, scale: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.down = down
        self.up = up
        self.scale = scale
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down, up = self.down.to(x.device), self.up.to(x.device)
        flat = x.reshape(-1, x.shape[-1])
        chunks = [((chunk.to(self.dtype) @ down.T) @ up.T * self.scale).to(x.dtype) for chunk in flat.split(_TOKEN_CHUNK)]
        return torch.cat(chunks).reshape(*x.shape[:-1], -1)


def moss_targets(layer: str, names: dict[str, str], dim: int) -> tuple[_Target, ...]:
    """Map an adapter's layer name to ``(module path, output rows)`` targets in the MOSS transformer.

    Args:
        layer: The layer name in the adapter file, with its format's prefix.
        names: Module paths by their underscore spelling (``path.replace(".", "_")``), covering every layer an
            adapter can name, including the fused ``layers.N.proj.qkv``.
        dim: The hidden size, which is the row count of each of q, k and v in a fused ``qkv``.

    Returns:
        The targets; empty for a layer that is not part of the model. A fused ``qkv`` gives q, k and v their
        thirds of the rows.
    """
    for prefix in _PREFIXES:
        if layer.startswith(prefix):
            layer = layer.removeprefix(prefix)
            break
    path = layer if "." in layer else names.get(layer)
    if path is None or names.get(path.replace(".", "_")) != path:
        return ()
    if not path.endswith(_FUSED_SUFFIX):
        return ((path, None),)
    base = path.removesuffix("qkv")
    return tuple((base + name, slice(i * dim, (i + 1) * dim)) for i, name in enumerate("qkv"))


def install_loras(transformer: torch.nn.Module, files: Sequence[LoraFile], *, device: torch.device | str) -> None:
    """Add the adapters in *files* to *transformer*'s layers, each at its strength, in order.

    Layers are wrapped once, so adapters from several files and several calls stack. A file's layers are all checked
    before any is applied. A file that applies nothing, or skips layers, gives a warning.

    Raises:
        ValueError: When a layer's factors do not fit the layer they map to (a LoRA made for a different model).
    """
    names, dim = _module_names(transformer)
    for file in files:
        applied, skipped = _install_file(transformer, file, names, dim, device)
        if skipped:
            warnings.warn(f"LoRA {Path(file.path).name}: {len(skipped)} layer(s) were skipped (first: {skipped[0]}).", stacklevel=2)
        if not applied:
            warn_unmatched_lora(file.path)


def _install_file(transformer: torch.nn.Module, file: LoraFile, names: dict[str, str], dim: int, device: torch.device | str) -> tuple[int, list[str]]:
    """Install one file's adapters; return how many layers were applied and the names of those skipped."""
    state = normalised_lora_state(file.path, _FAMILY, _LOADER)
    parts = split_lora_tensors(load_file(file.path) if state is None else state)
    adapters, skipped = _file_adapters(parts, _alpha_from_metadata(file.path), file, device)
    planned: list[tuple[_Adapter, list[tuple[str, slice | None]]]] = []
    for layer, adapter in adapters.items():
        targets = moss_targets(layer, names, dim)
        if not targets:
            skipped.append(layer)
            continue
        for path, rows in targets:
            check_fits("LoRA", layer, path, base_linear(transformer.get_submodule(path)), adapter.shape, parts=1 if rows is None else 3)
        planned.append((adapter, list(targets)))
    for adapter, targets in planned:
        for path, rows in targets:
            adapt_linear(transformer, path).terms.append(adapter.make(rows))
    return len(planned), skipped


def _module_names(transformer: torch.nn.Module) -> tuple[dict[str, str], int]:
    """Return the underscore-spelling lookup of the transformer's layers and its hidden size.

    The names come from the linear layers with any ``.base`` segment removed, so layers already wrapped by an earlier
    adapter stay findable. Each block's fused ``proj.qkv`` gets a virtual entry; the hidden size is a q layer's width.
    """
    paths = {".".join(part for part in name.split(".") if part != "base"): module for name, module in transformer.named_modules() if isinstance(module, torch.nn.Linear)}
    names = {path.replace(".", "_"): path for path in paths}
    dim = 0
    for path, module in paths.items():
        base = path.removesuffix(".q")
        if path.endswith(".proj.q") and f"{base}.k" in paths and f"{base}.v" in paths:
            fused = base + ".qkv"
            names[fused.replace(".", "_")] = fused
            dim = dim or module.out_features
    return names, dim


def _file_adapters(parts: LoraParts, metadata_scale: float | None, file: LoraFile, device: torch.device | str) -> tuple[dict[str, _Adapter], list[str]]:
    """Build the adapter of each supported layer in a split file; also return the names of the unsupported layers."""
    adapters: dict[str, _Adapter] = {}
    skipped = _unsupported_layers(parts)
    for layer, tensors in _by_layer(parts.lora).items():
        down = tensors.get("lora_down.weight", tensors.get("lora_A.weight"))
        up = tensors.get("lora_up.weight", tensors.get("lora_B.weight"))
        if down is None or up is None:
            continue  # bias or alpha tensors alone
        if _UNSUPPORTED_VARIANT_TENSORS & tensors.keys():
            skipped.append(layer)
            continue
        alpha = tensors.get("alpha")
        scale = float(alpha.item()) / down.shape[0] if alpha is not None else (metadata_scale if metadata_scale is not None else 1.0)
        adapters[layer] = _lora_adapter(down, up, scale * file.strength, file.dtype, device)
    for layer, tensors in parts.lokr.items():
        if _UNSUPPORTED_VARIANT_TENSORS & tensors.keys():
            skipped.append(layer)
            continue
        adapters[layer] = _lokr_adapter(tensors, file.strength, device)
    return adapters, skipped


def _unsupported_layers(parts: LoraParts) -> list[str]:
    """Return the layers of tensors that are neither LoRA nor LoKr (LoHa, ``diff`` deltas), without bare alphas."""
    layers = {split_key(key)[0] for key in parts.unsupported if split_key(key)[1] != "alpha"}
    return sorted(layers)


def _by_layer(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group tensors by layer name, keyed by tensor name inside each layer."""
    layers: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        layer, name = split_key(key)
        layers.setdefault(layer, {})[name] = tensor
    return layers


def _lora_adapter(down: torch.Tensor, up: torch.Tensor, scale: float, dtype: torch.dtype, device: torch.device | str) -> _Adapter:
    """Return the adapter for a LoRA layer; its factors move to *device* once, shared by every target."""
    down_d = down.detach().to(device=device, dtype=dtype)
    up_d = up.detach().to(device=device, dtype=dtype)

    def make(rows: slice | None) -> torch.nn.Module:
        return _LoraTerm(down_d, up_d if rows is None else up_d[rows].contiguous(), scale, dtype)

    return _Adapter((up.shape[0], down.shape[1]), make)


def _lokr_adapter(tensors: dict[str, torch.Tensor], strength: float, device: torch.device | str) -> _Adapter:
    """Return the adapter for a LoKr layer; its factors move to *device* as bfloat16 once, shared by every target."""
    w1, w2, scale = lokr_factors(tensors)
    w1_d = w1.detach().to(device=device, dtype=torch.bfloat16)
    w2_d = w2.detach().to(device=device, dtype=torch.bfloat16)

    def make(rows: slice | None) -> torch.nn.Module:
        return LoKrTerm(w1_d, w2_d, scale * strength, None if rows is None else range(rows.start, rows.stop))

    return _Adapter((w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]), make)


def _alpha_from_metadata(path: str) -> float | None:
    """Return ``lora_alpha / r`` from a file's ``lora_adapter_metadata`` (diffusers/PEFT), or None when it has none."""
    with safe_open(path, framework="pt") as handle:
        raw = (handle.metadata() or {}).get("lora_adapter_metadata")
    if raw is None:
        return None
    try:
        values: dict[str, Any] = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(values, dict):
        return None
    alpha, rank = _metadata_number(values, "lora_alpha"), _metadata_number(values, "r")
    return alpha / rank if alpha is not None and rank else None


def _metadata_number(values: dict[str, Any], name: str) -> float | None:
    """Return the number stored under *name*, bare or under a component prefix (``transformer.r``)."""
    for key, value in values.items():
        if (key == name or key.endswith(f".{name}")) and isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None
