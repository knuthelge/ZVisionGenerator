"""Ideogram 4 Instant's build settings: where the transformer and the shared components come from, and the build record.

Ideogram 4 Instant (fal's 8-step distilled Ideogram 4) has no ungated repo in a layout the backends load, so it is
built once into a model folder of its own: the transformer from a single BF16 file, the text encoder, VAE, tokenizer
and scheduler from the base model's repo. The ``ideogram4_instant`` config block pins every source.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.paths import get_ziv_data_dir, split_huggingface_revision

__all__ = [
    "BUILD_RECORD_NAME",
    "INSTANT_DIR_NAME",
    "TRANSFORMER_FORMATS",
    "InstantSource",
    "instant_source",
    "is_built",
    "is_instant_dir",
    "read_build_record",
    "write_build_record",
]

# The model folder's name in the models directory, the same as its alias: once built, the folder is the installed
# model ``ideo``, and its stored quants are ``ideo@q4`` and ``ideo@q8``.
INSTANT_DIR_NAME = "ideo"
# Written last into a finished build; a folder without it (or with an outdated one) is rebuilt.
BUILD_RECORD_NAME = "ziv-build.json"
# ``bf16``: diffusers' names and dtype (the CUDA backend). ``fp8``: the base FP8 repo's names and storage (mflux).
TRANSFORMER_FORMATS = ("bf16", "fp8")
_RECORD_VERSION = 1


@dataclass(frozen=True)
class InstantSource:
    """Where one platform's Ideogram 4 Instant build comes from.

    Attributes:
        transformer_repo: Hugging Face repo holding the single-file BF16 transformer.
        transformer_revision: The commit of *transformer_repo* to download.
        transformer_file: The transformer's file name in the repo.
        components: The base model's repo (``owner/repo@revision``) for the text encoder, VAE, tokenizer and scheduler.
        transformer_format: How the transformer is written, one of :data:`TRANSFORMER_FORMATS`.
    """

    transformer_repo: str
    transformer_revision: str
    transformer_file: str
    components: str
    transformer_format: str

    @property
    def components_repo(self) -> str:
        """Return the components repo id without its revision."""
        return split_huggingface_revision(self.components)[0]

    @property
    def components_revision(self) -> str | None:
        """Return the pinned revision of the components repo, if any."""
        return split_huggingface_revision(self.components)[1]


def instant_source(config: dict[str, Any], platform_key: str) -> InstantSource:
    """Return the Ideogram 4 Instant build source for *platform_key* from *config*'s ``ideogram4_instant`` block.

    Raises:
        ValueError: If the block is missing or incomplete, or has no entry for *platform_key*.
    """
    block = config.get("ideogram4_instant")
    if not isinstance(block, dict):
        raise ValueError("config 'ideogram4_instant' must be a mapping.")
    transformer = block.get("transformer")
    if not isinstance(transformer, dict) or not all(isinstance(transformer.get(key), str) and transformer[key] for key in ("repo", "revision", "filename")):
        raise ValueError("config 'ideogram4_instant.transformer' needs 'repo', 'revision' and 'filename'.")
    components = _platform_value(block, "components", platform_key)
    transformer_format = _platform_value(block, "transformer_format", platform_key)
    if transformer_format not in TRANSFORMER_FORMATS:
        raise ValueError(f"config 'ideogram4_instant.transformer_format.{platform_key}' must be one of {', '.join(TRANSFORMER_FORMATS)}.")
    return InstantSource(
        transformer_repo=transformer["repo"],
        transformer_revision=transformer["revision"],
        transformer_file=transformer["filename"],
        components=components,
        transformer_format=transformer_format,
    )


def _platform_value(block: dict[str, Any], key: str, platform_key: str) -> str:
    """Return ``block[key][platform_key]`` as a non-empty string."""
    values = block.get(key)
    value = values.get(platform_key) if isinstance(values, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"config 'ideogram4_instant.{key}' has no entry for '{platform_key}'.")
    return value


def is_instant_dir(path: str | Path) -> bool:
    """Return whether *path* is the Ideogram 4 Instant model folder in the models directory (built or not)."""
    candidate = Path(path).expanduser()
    return candidate.name == INSTANT_DIR_NAME and _same_path(candidate.parent, get_ziv_data_dir() / "models")


def _same_path(first: Path, second: Path) -> bool:
    """Return whether *first* and *second* are the same location."""
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return False


def _record(source: InstantSource) -> dict[str, Any]:
    """Return the build record of a folder built from *source*."""
    return {"version": _RECORD_VERSION, **asdict(source)}


def read_build_record(model_dir: Path) -> dict[str, Any] | None:
    """Return *model_dir*'s build record, or ``None`` when it has none or it cannot be read."""
    try:
        record = json.loads((model_dir / BUILD_RECORD_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return record if isinstance(record, dict) else None


def is_built(model_dir: Path, source: InstantSource) -> bool:
    """Return whether *model_dir* holds a finished build from exactly *source*."""
    return read_build_record(model_dir) == _record(source)


def write_build_record(model_dir: Path, source: InstantSource) -> None:
    """Mark *model_dir* as a finished build from *source*; written last."""
    (model_dir / BUILD_RECORD_NAME).write_text(json.dumps(_record(source), indent=2) + "\n", encoding="utf-8")
