"""Size the LoRAs a Web UI memory estimate covers, resolved as a job resolves them."""

from __future__ import annotations

from zvisiongenerator.utils.h3_memory import h3_lora_bytes
from zvisiongenerator.utils.lora import resolve_lora_references


def lora_bytes_for_fit(lora: str | None, *, family: str) -> int:
    """Return the total size of the LoRA files the ``name:weight,...`` list *lora* names, for a model of *family*.

    Only MiniMax H3 adds LoRA files to its memory estimate; any other family counts nothing, so an LTX estimate never
    depends on its LoRAs. The list resolves exactly as in a job (local names and paths; a folder is accepted).

    Raises:
        ValueError: If *lora* is malformed, a remote Hugging Face reference, or names a LoRA that is not there.
    """
    text = (lora or "").strip()
    if family != "h3" or not text:
        return 0
    paths, _weights = resolve_lora_references(text, require_file=False)
    return h3_lora_bytes(paths)
