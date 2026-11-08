"""Shared helpers for the image tests: model metadata with test pins for the models built on first use."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._factories import metadata_copy, use_model_metadata
from zvisiongenerator.utils.model_metadata import model_entry

__all__ = ["moss_sft_files", "pin_instant", "pin_moss"]

MOSS_SFT = "owner/sft@" + "a" * 40
INSTANT_COMPONENTS = "owner/base@" + "b" * 40


def moss_sft_files() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the file patterns and the ignore patterns of the SFT target of the packaged ``moss`` entry."""
    sft = next(target for target in model_entry("moss")["downloads"] if target["role"] == "sft")
    return tuple(sft["files"]), tuple(sft["ignore"])


def pin_moss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, sft: str = MOSS_SFT, lora_revision: str = "b" * 40) -> None:
    """Serve metadata whose ``moss`` and ``moss-sft`` entries pin the SFT repo *sft* and the Turbo LoRA at *lora_revision*."""
    data = metadata_copy()
    for model_id in ("moss", "moss-sft"):
        for target in data["models"][model_id]["downloads"]:
            if target.get("role") == "sft":
                target["ref"] = sft
            elif target.get("role") == "turbo_lora":
                target["ref"] = f"owner/lora@{lora_revision}"
                target["files"] = ["lora.safetensors"]
    use_model_metadata(monkeypatch, tmp_path, data)


def pin_instant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, components: str = INSTANT_COMPONENTS, transformer_format: str = "bf16", platforms: tuple[str, ...] = ("darwin", "win32", "linux")
) -> None:
    """Serve metadata whose ``ideo`` entry pins the components repo *components* and a transformer file, on *platforms*."""
    data = metadata_copy()
    entry: dict[str, Any] = data["models"]["ideo"]
    downloads = entry["downloads"]
    for platform_key in platforms:
        for target in downloads[platform_key]:
            if target["role"] == "components":
                target["ref"] = components
            else:
                target["ref"] = "owner/instant@" + "a" * 40
                target["files"] = ["instant.safetensors"]
    entry["transformer_format"] = dict.fromkeys(platforms, transformer_format)
    use_model_metadata(monkeypatch, tmp_path, data)
