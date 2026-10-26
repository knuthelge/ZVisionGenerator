"""MOSS-Image's build settings: where the SFT model and the Turbo LoRA come from, and the build records.

MOSS-Image is one model with two aliases. ``moss-sft`` is the SFT checkpoint as released; ``moss`` (Turbo) is the same
checkpoint run with a 4-step LoRA. Each alias is built once into a model folder of its own that links the downloaded
files; the Turbo folder also records its LoRA, which sits next to the SFT files. The ``moss_image`` config block pins
every source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.built_model import is_models_dir_child, read_build_record, write_record
from zvisiongenerator.utils.paths import split_huggingface_revision
from zvisiongenerator.utils.stored_quant import is_current, parse_stored_quant_name, stored_quant_dir

__all__ = [
    "LINKED_COMPONENTS",
    "SFT_DIR_NAME",
    "SFT_IGNORE",
    "SFT_PATTERNS",
    "TURBO_DIR_NAME",
    "TURBO_LORA_NAME",
    "MossSource",
    "build_record",
    "built_moss_base",
    "is_built",
    "is_moss_dir",
    "is_turbo_dir",
    "moss_base_dir",
    "moss_source",
    "shared_quant_source",
    "turbo_lora_file",
    "write_build_record",
]

# The model folders' names in the models directory, the same as their aliases: once built, a folder is the installed
# model, and its stored quants are ``moss@q4``, ``moss-sft@q8`` and so on.
TURBO_DIR_NAME = "moss"
SFT_DIR_NAME = "moss-sft"
# The Turbo LoRA's file name inside the Turbo folder.
TURBO_LORA_NAME = "turbo_lora.safetensors"
# The files of the SFT repo the build downloads; its pipeline script is never fetched or run.
SFT_PATTERNS = ("model_index.json", "LICENSE", "transformer/*", "text_encoder/*", "tokenizer/*", "scheduler/*", "vae/*")
SFT_IGNORE = ("*.py",)
# The SFT repo's folders a built model folder links as they are.
LINKED_COMPONENTS = ("transformer", "text_encoder", "tokenizer", "scheduler", "vae")
_RECORD_VERSION = 1


@dataclass(frozen=True)
class MossSource:
    """Where MOSS-Image's build comes from.

    Attributes:
        sft: The SFT repo (``owner/repo@revision``).
        turbo_repo: Hugging Face repo holding the Turbo LoRA.
        turbo_revision: The commit of *turbo_repo* to download.
        turbo_file: The LoRA's file name in *turbo_repo*.
    """

    sft: str
    turbo_repo: str
    turbo_revision: str
    turbo_file: str

    @property
    def sft_repo(self) -> str:
        """Return the SFT repo id without its revision."""
        return split_huggingface_revision(self.sft)[0]

    @property
    def sft_revision(self) -> str | None:
        """Return the pinned revision of the SFT repo."""
        return split_huggingface_revision(self.sft)[1]


def moss_source(config: dict[str, Any]) -> MossSource:
    """Return the MOSS-Image build source from *config*'s ``moss_image`` block.

    Raises:
        ValueError: If the block is missing or incomplete, or the SFT repo has no pinned revision.
    """
    block = config.get("moss_image")
    if not isinstance(block, dict):
        raise ValueError("config 'moss_image' must be a mapping.")
    sft = block.get("sft")
    if not isinstance(sft, str) or not sft or split_huggingface_revision(sft)[1] is None:
        raise ValueError("config 'moss_image.sft' must be a repo with a pinned revision ('owner/repo@revision').")
    turbo = block.get("turbo_lora")
    if not isinstance(turbo, dict) or not all(isinstance(turbo.get(key), str) and turbo[key] for key in ("repo", "revision", "filename")):
        raise ValueError("config 'moss_image.turbo_lora' needs 'repo', 'revision' and 'filename'.")
    return MossSource(sft=sft, turbo_repo=turbo["repo"], turbo_revision=turbo["revision"], turbo_file=turbo["filename"])


def is_moss_dir(path: str | Path) -> bool:
    """Return whether *path* is a MOSS-Image model folder in the models directory (built or not)."""
    return is_models_dir_child(path, (TURBO_DIR_NAME, SFT_DIR_NAME))


def is_turbo_dir(path: str | Path) -> bool:
    """Return whether *path* is the MOSS-Image Turbo model folder in the models directory (built or not)."""
    return is_moss_dir(path) and Path(path).expanduser().name == TURBO_DIR_NAME


def build_record(source: MossSource, *, turbo: bool) -> dict[str, Any]:
    """Return the build record of a folder built from *source*; only the Turbo folder records its LoRA."""
    record: dict[str, Any] = {"version": _RECORD_VERSION, "sft": source.sft}
    if turbo:
        record["turbo_lora"] = {"repo": source.turbo_repo, "revision": source.turbo_revision, "filename": source.turbo_file}
    return record


def is_built(model_dir: Path, source: MossSource) -> bool:
    """Return whether *model_dir* holds a finished build from exactly *source* (the folder name decides Turbo or SFT)."""
    return read_build_record(model_dir) == build_record(source, turbo=is_turbo_dir(model_dir))


def write_build_record(model_dir: Path, source: MossSource, *, turbo: bool) -> None:
    """Mark *model_dir* as a finished build from *source*; written last.

    Args:
        model_dir: The folder to mark.
        source: The pinned source it was built from.
        turbo: Whether the folder is the Turbo build, which records its LoRA.
    """
    write_record(model_dir, build_record(source, turbo=turbo))


def moss_base_dir(path: str | Path) -> Path:
    """Return the base folder of a MOSS-Image *path*: the path itself, or the folder a stored quant copy (``moss@q4``) belongs to.

    The result is not checked to be a MOSS-Image folder; see :func:`built_moss_base`.
    """
    candidate = Path(path).expanduser()
    parsed = parse_stored_quant_name(candidate.name)
    return candidate.with_name(parsed[0]) if parsed is not None else candidate


def turbo_lora_file(base: Path) -> Path | None:
    """Return the Turbo LoRA file in the base folder *base* (see :func:`moss_base_dir`), or ``None`` when it has none."""
    lora = base / TURBO_LORA_NAME
    return lora if lora.is_file() else None


def built_moss_base(path: str | Path) -> Path | None:
    """Return the ``moss`` / ``moss-sft`` folder in the models directory that *path* is, or is a stored copy of; else ``None``."""
    base = moss_base_dir(path)
    return base if is_moss_dir(base) else None


def shared_quant_source(source: Path, bits: int, backend_format: str) -> Path | None:
    """Return the other alias's stored copy at *bits* that can stand in for *source*'s own, or ``None``.

    Turbo and SFT share one transformer and text encoder, so a copy of either holds the weights of both. The other
    alias's copy qualifies when it exists, is current for its own source and both folders were built from the same
    SFT pin.
    """
    if not is_moss_dir(source):
        return None
    other = source.with_name(SFT_DIR_NAME if source.name == TURBO_DIR_NAME else TURBO_DIR_NAME)
    copy = stored_quant_dir(other, bits)
    if not copy.is_dir() or not other.is_dir() or not is_current(copy, other, bits, backend_format):
        return None
    own, theirs = read_build_record(source), read_build_record(other)
    if own is None or theirs is None or own.get("sft") != theirs.get("sft"):
        return None
    return copy
