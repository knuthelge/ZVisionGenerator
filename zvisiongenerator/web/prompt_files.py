"""Normalize, inspect, read, and atomically update host-local prompt files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

from zvisiongenerator.utils.prompts import PromptFileInspection, PromptFileOption, inspect_prompts_file, inspect_prompts_text


@dataclass(frozen=True)
class PromptFileDocument:
    """Represent a normalized prompt file plus its active option metadata."""

    path: str
    options: list[dict[str, str | int | None]]
    raw_text: str | None = None


def inspect_prompt_file(path: str, *, accepted_extensions: tuple[str, ...]) -> PromptFileDocument:
    """Inspect a host-local prompt file and return active option metadata."""
    normalized_path = normalize_prompt_file_path(path, accepted_extensions=accepted_extensions)
    inspection = inspect_prompts_file(str(normalized_path))
    return PromptFileDocument(path=str(normalized_path), options=_serialize_options(inspection.options))


def read_prompt_file(path: str, *, accepted_extensions: tuple[str, ...]) -> PromptFileDocument:
    """Read raw prompt-file YAML plus active option metadata."""
    normalized_path = normalize_prompt_file_path(path, accepted_extensions=accepted_extensions)
    raw_text = normalized_path.read_text(encoding="utf-8")
    inspection = inspect_prompts_text(raw_text, source_name=str(normalized_path))
    return PromptFileDocument(path=str(normalized_path), raw_text=raw_text, options=_serialize_options(inspection.options))


def write_prompt_file(path: str, raw_text: str, *, accepted_extensions: tuple[str, ...]) -> PromptFileDocument:
    """Validate and atomically replace a prompt file with raw YAML text."""
    normalized_path = normalize_prompt_file_path(path, accepted_extensions=accepted_extensions)
    inspection = inspect_prompts_text(raw_text, source_name=str(normalized_path))
    _write_atomic_text(normalized_path, raw_text)
    return PromptFileDocument(path=str(normalized_path), options=_serialize_options(inspection.options))


def resolve_prompt_file_option(path: str, option_id: str, *, accepted_extensions: tuple[str, ...]) -> tuple[str, PromptFileOption]:
    """Resolve one active prompt-file option by its stable id."""
    normalized_path = normalize_prompt_file_path(path, accepted_extensions=accepted_extensions)
    inspection = inspect_prompts_file(str(normalized_path))
    option = _find_option(inspection, option_id)
    return str(normalized_path), option


def normalize_prompt_file_path(path: str, *, accepted_extensions: tuple[str, ...]) -> Path:
    """Expand and validate a prompt-file path as a host-local existing file."""
    text = path.strip()
    if not text:
        raise ValueError("A prompt file path is required.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.suffix.lower() not in accepted_extensions:
        raise ValueError(f"Prompt file must use one of: {', '.join(accepted_extensions)}.")
    if not candidate.exists():
        raise ValueError(f"Prompt file does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Prompt file path must point to a file: {candidate}")
    return candidate


def _find_option(inspection: PromptFileInspection, option_id: str) -> PromptFileOption:
    for option in inspection.options:
        if option.id == option_id:
            return option
    raise ValueError(f"Prompt option '{option_id}' is missing or inactive.")


def _serialize_options(options: list[PromptFileOption]) -> list[dict[str, str | int | None]]:
    return [
        {
            "id": option.id,
            "set_name": option.set_name,
            "source_index": option.source_index,
            "label": _build_option_label(option),
            "prompt_preview": option.prompt,
            "negative_preview": option.negative_prompt,
        }
        for option in options
    ]


def _build_option_label(option: PromptFileOption) -> str:
    ordinal = option.source_index + 1
    excerpt = option.prompt.strip().replace("\n", " ")
    if len(excerpt) > 60:
        excerpt = f"{excerpt[:57].rstrip()}..."
    return f"{option.set_name} #{ordinal} · {excerpt}"


def _write_atomic_text(path: Path, raw_text: str) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(raw_text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
