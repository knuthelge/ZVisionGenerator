"""Route a user H3 LoRA by target for the folded 4-bit copy.

``utils/h3_lora`` reads every format (PEFT, ai-toolkit, kohya/musubi, Comfy-style fused qkv) into diffusers' module names under
the ``transformer.`` prefix and merges the turbo LoRA in fp32; its names are re-exported here. This module owns what follows:

* A user LoRA is split by target. ``adaln_proj`` has no linear at run time (its output is a folded table), so it becomes a
  table delta. ``token_refiner`` is merged into the bf16 refiner. Everything else goes to PEFT on the NF4 copy.
  A LoRA on the timestep embedding is refused: the folded tables are a function of it.

Everything here is pure; C11's ``load_copy`` applies the parts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zvisiongenerator.utils.h3_lora import PREFIX, TIMESTEP_MODULES, LoraFile, LoraMerger, check_targets, load_lora, load_pairs, merge, pairs_from_state

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
        if module.split(".", 1)[0] in TIMESTEP_MODULES:
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
