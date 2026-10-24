"""Load H3 LoRA files, merge the turbo LoRA in fp32 and route a user LoRA by target.

``utils/h3_lora`` reads every format (PEFT, ai-toolkit, kohya/musubi, Comfy-style fused qkv) into diffusers' module names under
the ``transformer.`` prefix, and its names are re-exported here. This module owns what follows:

* The turbo LoRA is merged into the bf16 weights with the maths in fp32. The config scale *replaces* ``alpha / rank``
  (it is never multiplied by it), so the loader hands the merger the raw factors with every ``.alpha`` dropped.
* A user LoRA is split by target. ``adaln_proj`` has no linear at run time (its output is a folded table), so it becomes a
  table delta. ``token_refiner`` is merged into the bf16 refiner. Everything else goes to PEFT on the NF4 copy.
  A LoRA on the timestep embedding is refused: the folded tables are a function of it.

Everything here is pure; C11's ``load_copy`` applies the parts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.utils.h3_lora import PREFIX, LoraFile, fetch_lora_state, load_lora, pairs_from_state

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from torch import nn

    Pair = tuple[torch.Tensor, torch.Tensor]
    Scaled = tuple[torch.Tensor, torch.Tensor, float]

__all__ = [
    "PREFIX",
    "LoraFile",
    "LoraMerger",
    "PeftPart",
    "UserLoraParts",
    "bake_scales",
    "check_targets",
    "load_lora",
    "load_pairs",
    "merge",
    "merge_refiner",
    "pairs_from_state",
    "peft_state",
    "split_user_lora",
]

_TIMESTEP_MODULES = ("time_embedder", "time_proj")
_ADALN_BLOCK = re.compile(r"transformer_blocks\.(\d+)\.adaln_proj\.linear")


@dataclass(frozen=True)
class PeftPart:
    """The modules of a user LoRA that PEFT applies to the NF4 copy.

    Attributes:
        state: Converted keys (``transformer.`` prefix) of the PEFT-routed modules.
        scales: ``{module: alpha / rank}`` for those modules, without the user weight. Feed :func:`peft_state` to PEFT.
        weight: The user weight, for ``set_adapters``.
    """

    state: dict[str, torch.Tensor]
    scales: dict[str, float]
    weight: float


@dataclass(frozen=True)
class UserLoraParts:
    """A user LoRA split by target.

    Attributes:
        adaln: ``{block index: (A, B, s)}`` for ``transformer_blocks.<n>.adaln_proj.linear``; ``s = weight * alpha / rank``.
        refiner: ``{module: (A, B, s)}`` for ``token_refiner.*``; ``s = weight * alpha / rank``.
        peft: Everything else.
        refused: Timestep-embedding modules (always empty after a strict split).
    """

    adaln: dict[int, Scaled]
    refiner: dict[str, Scaled]
    peft: PeftPart
    refused: tuple[str, ...]


def load_pairs(path: Path | str) -> dict[str, Pair]:
    """Load the raw ``{module: (lora_A, lora_B)}`` factors of the turbo LoRA, with every alpha ignored.

    Per-module ``.alpha`` tensors are dropped before the conversion (which would fold them into the factors) and a metadata alpha
    is not read, so the merge scale is the config's alone.
    """
    from safetensors.torch import load_file

    raw = {name: tensor for name, tensor in load_file(str(path)).items() if not name.endswith(".alpha")}
    return pairs_from_state(fetch_lora_state(raw)[0])


def check_targets(modules: Mapping[str, object] | list[str] | tuple[str, ...]) -> None:
    """Refuse a LoRA that targets the timestep embedding.

    Args:
        modules: Converted module names (or a mapping keyed by them).

    Raises:
        ValueError: If any module is under ``time_embedder`` or ``time_proj``.
    """
    refused = sorted(name for name in modules if name.split(".", 1)[0] in _TIMESTEP_MODULES)
    if refused:
        raise ValueError(f"LoRA targets the timestep embedding, which H3's folded AdaLN can't take ({', '.join(refused[:4])}).")


def merge(weight: torch.Tensor, down: torch.Tensor, up: torch.Tensor, scale: float, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return ``(weight + scale * up @ down)`` with the maths in fp32, cast to *out_dtype* (default: the weight's dtype).

    Raises:
        ValueError: If ``up @ down`` does not have the weight's shape.
    """
    if weight.ndim != 2 or weight.shape != (up.shape[0], down.shape[1]) or up.shape[1] != down.shape[0]:
        raise ValueError(f"LoRA factors up {tuple(up.shape)} @ down {tuple(down.shape)} do not fit a weight of shape {tuple(weight.shape)}.")
    delta = scale * (up.to(weight.device).float() @ down.to(weight.device).float())
    return (weight.float() + delta).to(out_dtype or weight.dtype)


class LoraMerger:
    """Merge one LoRA, at a fixed scale, into the weights of a copy as they stream past.

    The scale is the only factor: ``alpha / rank`` is not part of the input, so it cannot be applied twice.
    """

    def __init__(self, pairs: Mapping[str, Pair], scale: float, device: torch.device | str = "cpu") -> None:
        """Hold *pairs* to merge at *scale*.

        Raises:
            ValueError: If *scale* is not finite, or a pair targets the timestep embedding.
        """
        if not math.isfinite(scale):
            raise ValueError(f"LoRA merge scale must be finite, got {scale}.")
        check_targets(pairs)
        self.pairs = dict(pairs)
        self.scale = float(scale)
        self.device = device
        self.merged: list[str] = []

    def __contains__(self, module: str) -> bool:
        return module in self.pairs

    def apply(self, module: str, weight: torch.Tensor) -> torch.Tensor:
        """Return *weight* of *module* with the LoRA merged in (the weight's dtype), or *weight* itself when the LoRA does not target it."""
        if module not in self.pairs:
            return weight
        down, up = self.pairs[module]
        self.merged.append(module)
        try:
            return merge(weight.to(self.device), down, up, self.scale).to(device=weight.device, dtype=weight.dtype)
        except ValueError as exc:
            raise ValueError(f"LoRA target {module}: {exc}") from exc

    def unmerged(self) -> list[str]:
        """Return the LoRA modules no :meth:`apply` call has merged (a LoRA for a different model layout)."""
        return sorted(set(self.pairs) - set(self.merged))


def split_user_lora(
    state: Mapping[str, torch.Tensor],
    weight: float,
    *,
    scales: Mapping[str, float] | None = None,
    strict: bool = True,
) -> UserLoraParts:
    """Split a converted user LoRA by target (AdaLN table delta, refiner merge, PEFT).

    Args:
        state: The converted state dict (:attr:`LoraFile.state`).
        weight: The user's LoRA weight.
        scales: ``{module: alpha / rank}`` (:attr:`LoraFile.scales`); modules missing from it use 1.0.
        strict: Raise on a timestep-embedding target instead of listing it in ``refused``.

    Raises:
        ValueError: If *strict* and a module targets the timestep embedding.
    """
    pairs = pairs_from_state(state)
    per_module = scales or {}
    adaln: dict[int, Scaled] = {}
    refiner: dict[str, Scaled] = {}
    peft_modules: list[str] = []
    refused: list[str] = []
    for module in sorted(pairs):
        down, up = pairs[module]
        scale = float(per_module.get(module, 1.0))
        block = _ADALN_BLOCK.fullmatch(module)
        if module.split(".", 1)[0] in _TIMESTEP_MODULES:
            refused.append(module)
        elif block is not None:
            adaln[int(block.group(1))] = (down, up, weight * scale)
        elif module.startswith("token_refiner."):
            refiner[module] = (down, up, weight * scale)
        else:
            peft_modules.append(module)
    if refused and strict:
        check_targets(refused)
    peft = PeftPart(
        state={key: value for module in peft_modules for key, value in _module_items(state, module)},
        scales={module: float(per_module.get(module, 1.0)) for module in peft_modules},
        weight=float(weight),
    )
    return UserLoraParts(adaln=adaln, refiner=refiner, peft=peft, refused=tuple(refused))


def _module_items(state: Mapping[str, torch.Tensor], module: str) -> list[tuple[str, torch.Tensor]]:
    """Return the two ``(key, tensor)`` entries of *module* in *state*."""
    return [(key, state[key]) for key in (f"{PREFIX}{module}.lora_A.weight", f"{PREFIX}{module}.lora_B.weight")]


def bake_scales(state: Mapping[str, torch.Tensor], scales: Mapping[str, float]) -> dict[str, torch.Tensor]:
    """Return *state* with each module's ``alpha / rank`` multiplied into its ``lora_B``, so PEFT's own ``alpha == rank`` scaling of 1.0 is exact.

    Tensors of modules at scale 1.0 are returned as they are; no input tensor is modified.
    """
    result = dict(state)
    for module, scale in scales.items():
        if scale != 1.0:
            key = f"{PREFIX}{module}.lora_B.weight"
            result[key] = state[key] * scale
    return result


def peft_state(part: PeftPart) -> dict[str, torch.Tensor]:
    """Return the state dict to hand ``load_lora_adapter``: the PEFT part with its ``alpha / rank`` baked in (apply ``part.weight`` through ``set_adapters``)."""
    return bake_scales(part.state, part.scales)


def merge_refiner(model: nn.Module, refiner: Mapping[str, Scaled]) -> list[str]:
    """Merge the refiner part of a user LoRA into the bf16 refiner weights of *model*, in place.

    Args:
        model: The transformer; its ``token_refiner`` linears are floating-point.
        refiner: :attr:`UserLoraParts.refiner`.

    Returns:
        The merged module names.

    Raises:
        ValueError: If a module is missing from the model, is not a floating-point linear, or the factors do not fit it.
    """
    import torch

    merged: list[str] = []
    for module, (down, up, scale) in refiner.items():
        try:
            linear = model.get_submodule(module)
        except AttributeError as exc:
            raise ValueError(f"LoRA target {module} is not a module of this model.") from exc
        weight = getattr(linear, "weight", None)
        if weight is None or not weight.is_floating_point():
            raise ValueError(f"LoRA target {module} is not a floating-point linear; the refiner must stay unquantized.")
        with torch.no_grad():
            try:
                weight.copy_(merge(weight, down, up, scale))
            except ValueError as exc:
                raise ValueError(f"LoRA target {module}: {exc}") from exc
        merged.append(module)
    return merged
