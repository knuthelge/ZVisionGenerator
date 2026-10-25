"""Read MiniMax H3 LoRA files into diffusers' converted form and merge the turbo LoRA, on any platform.

``MiniMaxH3LoraLoaderMixin.lora_state_dict`` reads every format (PEFT, ai-toolkit, kohya/musubi, Comfy-style fused qkv) and
returns diffusers' module names under the ``transformer.`` prefix. This module wraps it and reads the converted state as
``{module: (lora_A, lora_B)}`` pairs. The turbo LoRA is merged into weights with the maths in fp32; its config scale
*replaces* ``alpha / rank`` (it is never multiplied by it), so the loader hands the merger the raw factors with every
``.alpha`` dropped. It holds no backend code: torch and diffusers are imported when a file is read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    Pair = tuple[torch.Tensor, torch.Tensor]

__all__ = ["PREFIX", "TIMESTEP_MODULES", "LoraFile", "LoraMerger", "check_targets", "fetch_lora_state", "load_lora", "load_pairs", "merge", "pairs_from_state"]

PREFIX = "transformer."
# The timestep embedding: the folded AdaLN tables are a function of it, so no LoRA may change it.
TIMESTEP_MODULES = ("time_embedder", "time_proj")
_PAIR_KEY = re.compile(rf"{re.escape(PREFIX)}(.+)\.lora_([AB])\.weight")


@dataclass(frozen=True)
class LoraFile:
    """A LoRA file in diffusers' converted form.

    Attributes:
        state: Keys ``transformer.<module>.lora_A.weight`` and ``.lora_B.weight``; any per-module ``.alpha`` is already folded into the factors.
        scales: ``{module: alpha / rank}`` from the file's metadata alpha, else 1.0 (the alpha-equals-rank convention). Never needed for a folded ``.alpha``.
    """

    state: dict[str, torch.Tensor]
    scales: dict[str, float]


def fetch_lora_state(source: str | dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, object] | None]:
    """Return ``(converted state, metadata)`` from diffusers' H3 loader; the one wrapper around it."""
    from diffusers.loaders import MiniMaxH3LoraLoaderMixin

    state, metadata = MiniMaxH3LoraLoaderMixin.lora_state_dict(source, return_lora_metadata=True)
    return state, metadata


def pairs_from_state(state: Mapping[str, torch.Tensor]) -> dict[str, Pair]:
    """Return ``{module: (lora_A, lora_B)}`` from a converted state dict.

    Raises:
        ValueError: If a key is not a ``transformer.`` LoRA weight, or a module has only one half of its pair.
    """
    halves: dict[str, dict[str, torch.Tensor]] = {}
    stray: list[str] = []
    for key, tensor in state.items():
        match = _PAIR_KEY.fullmatch(key)
        if match is None:
            stray.append(key)
            continue
        halves.setdefault(match.group(1), {})[match.group(2)] = tensor
    if stray:
        raise ValueError(f"LoRA state holds keys that are not transformer lora_A/lora_B weights (a LoRA for the reference partition?): {sorted(stray)[:4]}")
    incomplete = sorted(module for module, pair in halves.items() if len(pair) != 2)
    if incomplete:
        raise ValueError(f"LoRA module {incomplete[0]} has only one of lora_A and lora_B.")
    return {module: (pair["A"], pair["B"]) for module, pair in halves.items()}


def _metadata_scales(state: Mapping[str, torch.Tensor], metadata: Mapping[str, object] | None) -> dict[str, float]:
    """Return ``{module: alpha / rank}``: the metadata's uniform network alpha over each module's rank, else 1.0."""
    alpha = None if metadata is None else metadata.get(f"{PREFIX}lora_alpha")
    pairs = pairs_from_state(state)
    if not isinstance(alpha, int | float):
        return dict.fromkeys(pairs, 1.0)
    return {module: float(alpha) / int(down.shape[0]) for module, (down, _) in pairs.items()}


def load_lora(path: Path | str) -> LoraFile:
    """Load a user LoRA file (any format diffusers converts) with its effective ``alpha / rank`` per module."""
    state, metadata = fetch_lora_state(str(path))
    return LoraFile(state=state, scales=_metadata_scales(state, metadata))


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
    refused = sorted(name for name in modules if name.split(".", 1)[0] in TIMESTEP_MODULES)
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
