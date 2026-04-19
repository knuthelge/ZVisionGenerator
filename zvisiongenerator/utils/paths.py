"""Resolve model and LoRA paths from the ZIV data directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ziv_dirs_created: set[str] = set()
_AliasPlatformValue = str | dict[str, str]
_AliasMap = dict[str, str | dict[str, _AliasPlatformValue]]


def get_ziv_data_dir() -> Path:
    """Return the ZIV data directory (~/.ziv/ or ZIV_DATA_DIR).
    Creates the directory and models/ + loras/ subdirs if they don't exist.
    """
    env = os.environ.get("ZIV_DATA_DIR", "").strip()
    if env:
        data_dir = Path(env)
    else:
        data_dir = Path.home() / ".ziv"

    key = str(data_dir)
    if key not in _ziv_dirs_created:
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        (data_dir / "loras").mkdir(parents=True, exist_ok=True)
        _ziv_dirs_created.add(key)
    return data_dir


def resolve_model_path(name_or_path: str, *, aliases: _AliasMap | None = None, platform_labels: dict[str, str] | None = None) -> str:
    """Resolve a model name/path to a filesystem path.

    Resolution order:
    1. If name_or_path is absolute or contains '/' or '\\' → return as-is
    2. If bare name → check ~/.ziv/models/<name>/ → return if exists
     3. If aliases provided and name matches → return alias target
         - String alias: return directly
         - Dict alias: pick value for current platform (sys.platform)
    4. Otherwise → return as-is (assumed HuggingFace repo ID)
    """
    if os.path.isabs(name_or_path) or "/" in name_or_path or "\\" in name_or_path:
        return name_or_path

    candidate = get_ziv_data_dir() / "models" / name_or_path
    if candidate.is_dir():
        return str(candidate)

    if aliases and name_or_path in aliases:
        target = aliases[name_or_path]
        if isinstance(target, dict):
            value = target.get(sys.platform)
            if isinstance(value, str):
                return value
            if platform_labels is None:
                from zvisiongenerator.utils.config import load_config
                from zvisiongenerator.utils.platform import get_all_platform_labels

                platform_labels = get_all_platform_labels(load_config())
            available_platforms = [k for k, v in target.items() if isinstance(v, str)]
            available = ", ".join(platform_labels.get(k, k) for k in sorted(available_platforms))
            current = platform_labels.get(sys.platform, sys.platform)
            alias_message = value.get("message", "") if isinstance(value, dict) else ""
            parts = [f"'{name_or_path}' is not available on {current}.", f"Available on: {available}."]
            if alias_message:
                parts.append(alias_message)
            raise ValueError(" ".join(parts))
        return target

    return name_or_path


def resolve_lora_path(name_or_path: str) -> str:
    """Resolve a LoRA name/path to a filesystem path.

    Resolution order:
    1. If name_or_path is absolute or contains '/' or '\\' → return as-is
    2. If bare name → check ~/.ziv/loras/<name>.safetensors → return if exists
    3. If bare name → check ~/.ziv/loras/<name> (no extension) → return if exists
    4. Otherwise → return as-is
    """
    if os.path.isabs(name_or_path) or "/" in name_or_path or "\\" in name_or_path:
        return name_or_path

    data_dir = get_ziv_data_dir()

    candidate = data_dir / "loras" / f"{name_or_path}.safetensors"
    if candidate.is_file():
        return str(candidate)

    candidate = data_dir / "loras" / name_or_path
    if candidate.is_file():
        return str(candidate)

    return name_or_path
