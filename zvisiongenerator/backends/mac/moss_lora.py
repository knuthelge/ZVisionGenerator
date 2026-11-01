"""The LoRA mapping MOSS-Image's MLX transformer (mflux) loads user LoRAs and the Turbo LoRA through.

mflux's own mapping reads the six projections of each DiT layer under four prefixes. MOSS-Image LoRAs also come with
the names the Windows and Linux loader takes (see ``backends/cuda/image/moss_lora.py``): ComfyUI's
``model.diffusion_model.`` prefix, Kohya's flat ``lora_unet_`` names, a fused ``proj.qkv`` adapter, and adapters on the
text embedder, the timestep embedding and the input and output layers. The mapping is built from the family's module
list in :data:`~zvisiongenerator.utils.lora_keys.LORA_MODULES`, so every layout applies unmerged under its own names.
LoKr files load through the same targets.

mflux cannot apply a few things a file may hold, so :func:`prepare_lora` sorts them out first, as the Windows and Linux
loader does: a fused ``qkv`` LoKr cannot be split into three LoKr layers (the Kronecker product's rows do not divide
into thirds), so it is applied by :func:`apply_fused_lokr`; bare ``alpha`` tensors without matrices are dropped; LoHa,
full ``diff`` weights and DoRA scales are skipped with a warning.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlx import nn

from zvisiongenerator.utils.lora_keys import LORA_MODULES

if TYPE_CHECKING:
    import mlx.core as mx
    from mflux.models.common.lora.mapping.lora_mapping import LoRATarget

__all__ = ["PreparedLora", "apply_fused_lokr", "moss_lora_mapping", "needs_preparing", "prepare_lora"]

# Prefixes before the module path, as the trainers write them (the CUDA loader reads the same).
_PREFIXES = ("", "diffusion_model.", "model.diffusion_model.", "transformer.", "base_model.model.")
_KOHYA_PREFIX = "lora_unet_"
# The DiT's fused qkv, which the MLX transformer holds as separate q, k and v projections.
_FUSED_QKV = "layers.{n}.proj.qkv"
_QKV = ("q", "k", "v")
# A fused qkv tensor of a DiT layer, under any prefix: its layer number and the tensor's name.
_FUSED_TENSOR = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.proj\.qkv\.(?P<tensor>[\w.]+)$")
# Tensors mflux's LoRA layers cannot apply: LoHa factors, full weight deltas and DoRA scales.
_UNSUPPORTED = re.compile(r"\.(?:hada_\w+|diff|diff_b|dora_scale)$")


@dataclass(frozen=True)
class PreparedLora:
    """A LoRA file's tensors sorted for the loaders.

    Attributes:
        rest: The tensors mflux applies through :func:`moss_lora_mapping`.
        fused_lokr: The fused ``qkv`` LoKr tensors by layer number, for :func:`apply_fused_lokr`.
    """

    rest: dict[str, Any]
    fused_lokr: dict[int, dict[str, Any]]


def moss_lora_mapping() -> list[LoRATarget]:
    """Return the targets of every MOSS-Image linear a LoRA or LoKr file may adapt, in each naming the trainers use."""
    targets = []
    for template in LORA_MODULES["moss"]:
        if template == _FUSED_QKV:
            targets += [_fused_qkv_target(index, name) for index, name in enumerate(_QKV)]
        else:
            targets.append(_target(template.replace("{n}", "{block}")))
    return targets


def needs_preparing(names: list[str]) -> bool:
    """Return whether a LoRA file with tensors *names* holds anything :func:`prepare_lora` must take out for mflux."""
    return any(_is_fused_lokr(name) or _UNSUPPORTED.search(name) for name in names) or bool(_bare_alphas(names))


def prepare_lora(weights: dict[str, mx.array], label: str) -> PreparedLora:
    """Sort a LoRA file's tensors: fused ``qkv`` LoKr layers apart, bare alphas dropped, unsupported tensors skipped.

    Args:
        weights: The file's tensors by name.
        label: Names the file in the warning about skipped tensors.
    """
    fused_layers = {int(match.group("layer")) for name in weights if (match := _FUSED_TENSOR.search(name)) and _is_fused_lokr(name)}
    fused: dict[int, dict[str, Any]] = {}
    rest: dict[str, Any] = {}
    bare = _bare_alphas(list(weights))
    unsupported = {name for name in weights if _UNSUPPORTED.search(name)}
    for name, value in weights.items():
        match = _FUSED_TENSOR.search(name)
        if match is not None and int(match.group("layer")) in fused_layers:
            fused.setdefault(int(match.group("layer")), {})[match.group("tensor")] = value
        elif name not in bare and name not in unsupported:
            rest[name] = value
    if unsupported:
        warnings.warn(f"LoRA {label}: skipped {len(unsupported)} LoHa, diff or DoRA tensors this platform does not apply (e.g. {min(unsupported)}).", stacklevel=2)
    return PreparedLora(rest=rest, fused_lokr=fused)


def apply_fused_lokr(transformer: Any, fused: dict[int, dict[str, mx.array]], strength: float) -> None:
    """Wrap q, k and v of each layer in *fused* with its rows of the LoKr update at *strength*.

    Raises:
        ValueError: If a layer is not in the transformer or the update does not have three projections' rows.
    """
    from mflux.models.common.lora.mapping.lokr_factors import rebuild_lokr_factors

    dim = transformer.dim  # q, k and v are dim x dim; the layers may already be wrapped by other LoRAs
    for layer, tensors in fused.items():
        if layer >= len(transformer.layers):
            raise ValueError(f"The fused qkv LoKr names layer {layer}, which this model does not have.")
        w1, w2, alpha_scale = rebuild_lokr_factors(tensors)
        if w1.shape[0] * w2.shape[0] != 3 * dim:
            raise ValueError(f"The fused qkv LoKr of layer {layer} has {w1.shape[0] * w2.shape[0]} rows, not 3 x {dim}.")
        update = _FusedUpdate(w1, w2, strength * alpha_scale)
        projections = transformer.layers[layer].proj
        for index, name in enumerate(_QKV):
            projections[name] = _FusedLoKr(projections[name], update, slice(index * dim, (index + 1) * dim), last=index == len(_QKV) - 1)


class _FusedUpdate:
    """One layer's fused ``qkv`` LoKr update, computed once per input and shared by its q, k and v.

    The product is computed as ``w1 @ X @ w2ᵀ`` without building the Kronecker matrix. q, k and v see the same input in
    turn, so the result is kept from q's call until v's.
    """

    def __init__(self, w1: mx.array, w2: mx.array, scale: float) -> None:
        self.w1 = w1
        self.w2 = w2
        self.scale = scale
        self._input: mx.array | None = None
        self._result: mx.array | None = None

    def rows(self, x: mx.array, rows: slice, *, last: bool) -> mx.array:
        """Return *rows* of the scaled update for *x*; *last* drops the kept result once the last projection has it."""
        import mlx.core as mx

        if x is not self._input:
            groups = x.reshape(*x.shape[:-1], self.w1.shape[1], self.w2.shape[1])
            product = mx.matmul(mx.matmul(self.w1.astype(x.dtype), groups), self.w2.astype(x.dtype).T)
            self._input, self._result = x, self.scale * product.reshape(*x.shape[:-1], -1)
        result = self._result[..., rows]
        if last:
            self._input = self._result = None
        return result


class _FusedLoKr(nn.Module):
    """A projection of a fused ``qkv`` plus its rows of the layer's LoKr update."""

    def __init__(self, base: Any, update: _FusedUpdate, rows: slice, *, last: bool) -> None:
        super().__init__()
        self.base = base
        self._update = update
        self._rows = rows
        self._last = last

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + self._update.rows(x, self._rows, last=self._last)


def _target(model_path: str) -> LoRATarget:
    """Return the target of the linear at *model_path*, read under every prefix and as a Kohya flat name."""
    from mflux.models.common.lora.mapping.lora_mapping import LoRATarget

    paths = _source_paths(model_path)
    return LoRATarget(
        model_path=model_path,
        possible_up_patterns=_matrix_patterns(paths, ("lora_B", "lora_up")),
        possible_down_patterns=_matrix_patterns(paths, ("lora_A", "lora_down")),
        possible_alpha_patterns=[f"{path}.alpha" for path in paths],
        possible_lokr_w1_patterns=[f"{path}.lokr_w1" for path in paths],
        possible_lokr_w2_patterns=[f"{path}.lokr_w2" for path in paths],
    )


def _fused_qkv_target(index: int, name: str) -> LoRATarget:
    """Return the target of projection *name* read from a fused ``proj.qkv`` LoRA: its third of the up matrix rows.

    The down matrix and alpha are shared by the three projections, as in the fused layer.
    """
    from mflux.models.common.lora.mapping.lora_mapping import LoRATarget

    paths = _source_paths(_FUSED_QKV.replace("{n}", "{block}"))
    return LoRATarget(
        model_path=f"layers.{{block}}.proj.{name}",
        possible_up_patterns=_matrix_patterns(paths, ("lora_B", "lora_up")),
        possible_down_patterns=_matrix_patterns(paths, ("lora_A", "lora_down")),
        possible_alpha_patterns=[f"{path}.alpha" for path in paths],
        up_transform=_row_third(index),
    )


def _row_third(index: int) -> Callable[[mx.array], mx.array]:
    """Return a transform that keeps row third *index* of a fused ``(3 * dim, rank)`` up matrix."""

    def transform(up: mx.array) -> mx.array:
        rows = up.shape[0] // 3
        return up[index * rows : (index + 1) * rows]

    return transform


def _source_paths(model_path: str) -> list[str]:
    """Return the names a file may give the module at *model_path*: under each prefix, and Kohya's flat name."""
    return [f"{prefix}{model_path}" for prefix in _PREFIXES] + [_KOHYA_PREFIX + model_path.replace(".", "_")]


def _matrix_patterns(paths: list[str], names: tuple[str, str]) -> list[str]:
    """Return the LoRA matrix names under *paths*, with and without PEFT's ``default`` adapter name.

    mflux also matches each pattern without its ``.weight`` suffix.
    """
    return [f"{path}.{name}{adapter}.weight" for path in paths for name in names for adapter in ("", ".default")]


def _is_fused_lokr(name: str) -> bool:
    """Return whether *name* is a LoKr factor of a DiT layer's fused ``qkv``."""
    match = _FUSED_TENSOR.search(name)
    return match is not None and match.group("tensor").startswith("lokr_")


def _bare_alphas(names: list[str]) -> set[str]:
    """Return the ``alpha`` tensors whose layer has no other tensor in the file."""
    prefixes = {".".join(parts[:end]) for name in names if not name.endswith(".alpha") for parts in [name.split(".")] for end in range(1, len(parts))}
    return {name for name in names if name.endswith(".alpha") and name.removesuffix(".alpha") not in prefixes}
