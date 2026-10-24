"""Read MiniMax H3 LoRA files into diffusers' converted form, on any platform.

``MiniMaxH3LoraLoaderMixin.lora_state_dict`` reads every format (PEFT, ai-toolkit, kohya/musubi, Comfy-style fused qkv) and
returns diffusers' module names under the ``transformer.`` prefix. This module wraps it and reads the converted state as
``{module: (lora_A, lora_B)}`` pairs. It holds no backend code: torch and diffusers are imported when a file is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    Pair = tuple[torch.Tensor, torch.Tensor]

__all__ = ["PREFIX", "LoraFile", "fetch_lora_state", "load_lora", "pairs_from_state"]

PREFIX = "transformer."
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
