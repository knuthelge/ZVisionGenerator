"""LoRA CLI argument parsing utilities."""

from __future__ import annotations


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
