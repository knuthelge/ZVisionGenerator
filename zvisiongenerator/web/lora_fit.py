"""Size the LoRAs a Web UI memory estimate covers, resolved as a job resolves them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from zvisiongenerator.utils.h3_memory import h3_lora_bytes
from zvisiongenerator.utils.lora import parse_lora_arg, resolve_lora_references
from zvisiongenerator.utils.paths import resolve_lora_path


def lora_bytes_for_fit(lora: str | None, *, family: str, downloading: Mapping[str, int] | None = None) -> int:
    """Return the total size of the LoRA files the ``name:weight,...`` list *lora* names, for a model of *family*.

    Only MiniMax H3 adds LoRA files to its memory estimate; any other family counts nothing, so an LTX estimate never
    depends on its LoRAs. The list resolves exactly as in a job (local names and paths; a folder is accepted). A LoRA
    whose file is not there yet but is downloading counts the size the download expects.

    Args:
        lora: The ``name:weight,...`` list.
        family: The video model's family.
        downloading: ``{name: expected size in bytes}`` of the LoRAs being downloaded (the curated entry's size or the Hub's).

    Raises:
        ValueError: If *lora* is malformed, a remote Hugging Face reference, or names a LoRA that is not there and is not downloading.
    """
    text = (lora or "").strip()
    if family != "h3" or not text:
        return 0
    expected = 0
    resolvable = []
    for name, weight in parse_lora_arg(text):
        if name in (downloading or {}) and not Path(resolve_lora_path(name)).exists():
            expected += (downloading or {})[name]
        else:
            resolvable.append(f"{name}:{weight}")
    if not resolvable:
        return expected
    paths, _weights = resolve_lora_references(",".join(resolvable), require_file=False)
    return expected + h3_lora_bytes(paths)
