"""Size the LoRAs a Web UI memory estimate covers, resolved as a job resolves them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.lora_size import lora_files_bytes
from zvisiongenerator.utils.paths import is_remote_lora_reference, resolve_lora_path

__all__ = ["LoraSize", "lora_size_for_fit"]


@dataclass(frozen=True)
class LoraSize:
    """What a LoRA list adds to a memory estimate: the bytes of the files it could size, and the LoRAs it could not."""

    total: int = 0
    unavailable: tuple[str, ...] = ()


def lora_size_for_fit(lora: str | None, downloading: Mapping[str, int] | None = None) -> LoraSize:
    """Return the size of the LoRA files the ``name:weight,...`` list *lora* names, and the names of those that are not available.

    A name resolves as in a job (local names and paths; a folder is accepted). Which models add the files to their estimate is the
    engine's decision. A LoRA whose file is not there yet but is downloading counts the size the download expects. A LoRA that is not there, or that is a remote Hugging Face reference (a job refuses it), counts 0
    bytes and is listed in ``unavailable`` by its file name, so the estimate still judges the job.

    Args:
        lora: The ``name:weight,...`` list.
        downloading: ``{name: expected size in bytes}`` of the LoRAs being downloaded (the curated entry's size or the Hub's).

    Raises:
        ValueError: If *lora* is malformed.
    """
    text = (lora or "").strip()
    if not text:
        return LoraSize()
    total = 0
    unavailable = []
    for name, _weight in parse_lora_arg(text):
        path = resolve_lora_path(name)
        if name in (downloading or {}) and not Path(path).exists():
            total += (downloading or {})[name]
        elif is_remote_lora_reference(name) or not Path(path).exists():
            unavailable.append(Path(name).name)
        else:
            total += lora_files_bytes([path])
    return LoraSize(total, tuple(unavailable))
