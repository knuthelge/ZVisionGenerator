"""Prompt file loading — extracted from cli.py."""

from __future__ import annotations

from typing import Any

import yaml

from zvisiongenerator.utils.prompt_compose import flatten_value, resolve_snippets


def load_prompts_file(path: str) -> dict[str, list[tuple[str, str | None]]]:
    """Load a prompts YAML file and return structured prompt data.

    Args:
        path: Path to the YAML prompts file.

    Returns:
        Dict mapping set names to lists of ``(prompt, negative_prompt)`` tuples.
        Inactive entries (``active: false``) are filtered out.
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse prompts file '{path}': {e}") from e

    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError(f"Prompt file is empty or malformed: {path}")

    snippets = raw_data.pop("snippets", None) or {}

    if not isinstance(snippets, dict):
        raise ValueError(f"'snippets' block in {path} must be a mapping, got {type(snippets).__name__}.")

    active_data: dict[str, list[dict[str, Any]]] = {}
    for set_name, prompts in raw_data.items():
        if not isinstance(prompts, list):
            raise ValueError(f"Prompt set '{set_name}' in {path} must be a list of mappings, got {type(prompts).__name__}.")
        for i, entry in enumerate(prompts):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {i} in prompt set '{set_name}' in {path} must be a mapping with at least a 'prompt' key, got {type(entry).__name__}.")
            if "prompt" not in entry:
                raise ValueError(f"Entry {i} in prompt set '{set_name}' in {path} is missing the required 'prompt' key.")
        active_data[set_name] = [p for p in prompts if p.get("active", True)]

    prompts_data: dict[str, list[tuple[str, str | None]]] = {}
    for set_name, entries in active_data.items():
        if not entries:
            continue
        pairs: list[tuple[str, str | None]] = []
        for entry in entries:
            raw_prompt = entry.get("prompt")
            raw_negative = entry.get("negative")
            prompt_str = flatten_value(resolve_snippets(raw_prompt, snippets))
            if not prompt_str or not prompt_str.strip():
                raise ValueError(f"Entry in prompt set '{set_name}' in {path} has an empty prompt after snippet resolution.")
            neg_str = flatten_value(resolve_snippets(raw_negative, snippets)) or None
            pairs.append((prompt_str, neg_str))
        prompts_data[set_name] = pairs

    return prompts_data
