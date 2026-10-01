"""LoRA CLI argument parsing and reference validation utilities."""

from __future__ import annotations

import os

from zvisiongenerator.utils.paths import is_remote_lora_reference, resolve_lora_path


def parse_lora_arg(value: str) -> list[tuple[str, float]]:
    """Parse '--lora name1:0.8,name2:0.5' into [(name, weight), ...].

    Each comma-separated entry is 'name' or 'name:weight'.
    Weight defaults to 1.0 when omitted.

    Raises:
        ValueError: On empty name or non-numeric weight.
    """
    result = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            raise ValueError("Empty LoRA entry in --lora value")
        idx = entry.rfind(":")
        if idx != -1:
            maybe_weight = entry[idx + 1 :].strip()
            if maybe_weight:
                try:
                    weight = float(maybe_weight)
                    name = entry[:idx].strip()
                except ValueError:
                    name = entry
                    weight = 1.0
            else:
                name = entry[:idx].strip()
                weight = 1.0
        else:
            name = entry
            weight = 1.0
        if not name:
            raise ValueError(f"Empty LoRA name in --lora value: '{entry}'")
        result.append((name, weight))
    return result


def resolve_lora_references(value: str, *, require_file: bool) -> tuple[list[str], list[float]]:
    """Parse a ``--lora`` value and resolve it to loadable local paths and weights.

    Shared by the image/video CLIs and the Web UI so their validation cannot drift.

    Args:
        value: Raw ``name:weight,...`` specifier.
        require_file: True when the backend needs a LoRA file (image backends);
            False also accepts a directory (the diffusers video backend).

    Raises:
        ValueError: Malformed value, remote HuggingFace reference, or missing LoRA.
    """
    parsed = parse_lora_arg(value)
    remote_loras = [name for name, _ in parsed if is_remote_lora_reference(name)]
    if remote_loras:
        raise ValueError(f"Remote HuggingFace LoRA references are not supported: {', '.join(remote_loras)}. Import the LoRA locally or pass a local LoRA path.")
    lora_paths = [resolve_lora_path(name) for name, _ in parsed]
    for path in lora_paths:
        if not (os.path.isfile(path) if require_file else os.path.exists(path)):
            raise ValueError(f"LoRA file not found: {path}")
    return lora_paths, [weight for _, weight in parsed]
