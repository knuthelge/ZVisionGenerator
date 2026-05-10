"""Load and inspect prompt files using the shared CLI YAML semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from zvisiongenerator.utils.prompt_compose import flatten_value, resolve_snippets


@dataclass(frozen=True)
class PromptFileOption:
    """Represent one active prompt option from a prompt file."""

    set_name: str
    source_index: int
    prompt: str
    negative_prompt: str | None

    @property
    def id(self) -> str:
        """Return the stable option identity for UI and API selection."""
        return f"{self.set_name}:{self.source_index}"


@dataclass(frozen=True)
class PromptFileInspection:
    """Hold the active prompt pairs and option metadata for a prompt file."""

    prompts_data: dict[str, list[tuple[str, str | None]]]
    options: list[PromptFileOption]


def load_prompts_file(path: str) -> dict[str, list[tuple[str, str | None]]]:
    """Load a prompts YAML file and return structured prompt data.

    Args:
        path: Path to the YAML prompts file.

    Returns:
        Dict mapping set names to lists of ``(prompt, negative_prompt)`` tuples.
        Inactive entries (``active: false``) are filtered out.
    """
    return inspect_prompts_file(path).prompts_data


def inspect_prompts_file(path: str) -> PromptFileInspection:
    """Load a prompt file and expose both prompt pairs and active option metadata."""
    raw_text = Path(path).read_text(encoding="utf-8")
    return inspect_prompts_text(raw_text, source_name=path)


def inspect_prompts_text(raw_text: str, *, source_name: str) -> PromptFileInspection:
    """Inspect raw prompt-file YAML text using the shared CLI parsing semantics."""
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse prompts file '{source_name}': {exc}") from exc

    raw_sets, snippets = _parse_prompt_mapping(raw_data, source_name)

    prompts_data: dict[str, list[tuple[str, str | None]]] = {}
    options: list[PromptFileOption] = []
    for set_name, entries in raw_sets.items():
        pairs: list[tuple[str, str | None]] = []
        for source_index, entry in enumerate(entries):
            if not entry.get("active", True):
                continue
            prompt_text, negative_prompt = _resolve_prompt_entry(entry, snippets, source_name=source_name, set_name=set_name)
            pairs.append((prompt_text, negative_prompt))
            options.append(
                PromptFileOption(
                    set_name=set_name,
                    source_index=source_index,
                    prompt=prompt_text,
                    negative_prompt=negative_prompt,
                )
            )
        if pairs:
            prompts_data[set_name] = pairs

    return PromptFileInspection(prompts_data=prompts_data, options=options)


def _parse_prompt_mapping(raw_data: Any, source_name: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError(f"Prompt file is empty or malformed: {source_name}")

    snippets = raw_data.get("snippets") or {}
    if not isinstance(snippets, dict):
        raise ValueError(f"'snippets' block in {source_name} must be a mapping, got {type(snippets).__name__}.")

    raw_sets: dict[str, list[dict[str, Any]]] = {}
    for set_name, prompts in raw_data.items():
        if set_name == "snippets":
            continue
        if not isinstance(prompts, list):
            raise ValueError(f"Prompt set '{set_name}' in {source_name} must be a list of mappings, got {type(prompts).__name__}.")
        validated_entries: list[dict[str, Any]] = []
        for index, entry in enumerate(prompts):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {index} in prompt set '{set_name}' in {source_name} must be a mapping with at least a 'prompt' key, got {type(entry).__name__}.")
            if "prompt" not in entry:
                raise ValueError(f"Entry {index} in prompt set '{set_name}' in {source_name} is missing the required 'prompt' key.")
            validated_entries.append(entry)
        raw_sets[set_name] = validated_entries
    return raw_sets, snippets


def _resolve_prompt_entry(
    entry: dict[str, Any],
    snippets: dict[str, Any],
    *,
    source_name: str,
    set_name: str,
) -> tuple[str, str | None]:
    raw_prompt = entry.get("prompt")
    raw_negative = entry.get("negative")
    prompt_text = flatten_value(resolve_snippets(raw_prompt, snippets))
    if not prompt_text or not prompt_text.strip():
        raise ValueError(f"Entry in prompt set '{set_name}' in {source_name} has an empty prompt after snippet resolution.")
    negative_prompt = flatten_value(resolve_snippets(raw_negative, snippets)) or None
    return prompt_text, negative_prompt
