"""Ideogram 4 Instant's build settings: where the transformer and the shared components come from, and the build record.

Ideogram 4 Instant (fal's 8-step distilled Ideogram 4) has no ungated repo in a layout the backends load, so it is
built once into a model folder of its own: the transformer from a single BF16 file, the text encoder, VAE, tokenizer
and scheduler from the base model's repo. The ``ideo`` entry of the model metadata pins every source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from zvisiongenerator.downloads.declaration import model_targets, role_targets
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.built_model import is_models_dir_child, read_build_record, write_record
from zvisiongenerator.utils.model_metadata import for_platform, model_entry
from zvisiongenerator.utils.paths import join_huggingface_revision, split_huggingface_revision

__all__ = [
    "INSTANT_DIR_NAME",
    "LICENSE_FILES",
    "LINKED_COMPONENTS",
    "TRANSFORMER_FORMATS",
    "InstantSource",
    "instant_downloads_dir",
    "instant_source",
    "instant_targets",
    "is_built",
    "is_instant_dir",
    "write_build_record",
]

# The model folder's name in the models directory, the same as its alias: once built, the folder is the installed
# model ``ideo``, and its stored quants are ``ideo@q4`` and ``ideo@q8``.
INSTANT_DIR_NAME = "ideo"
# ``bf16``: diffusers' names and dtype (the CUDA backend). ``fp8``: the base FP8 repo's names and storage (mflux).
TRANSFORMER_FORMATS = ("bf16", "fp8")
_RECORD_VERSION = 1
# The base repo's components the build reuses as they are.
LINKED_COMPONENTS = ("scheduler", "text_encoder", "tokenizer", "vae")
# Ideogram's license agreement and attribution notice travel with the model.
LICENSE_FILES = ("LICENSE.md", "NOTICE")


@dataclass(frozen=True)
class InstantSource:
    """Where one platform's Ideogram 4 Instant build comes from; its fields are the build record.

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


def instant_targets(platform_key: str) -> tuple[DownloadTarget, DownloadTarget]:
    """Return the ``components`` and ``transformer`` targets of the ``ideo`` entry of the model metadata for *platform_key*."""
    pairs = model_targets(INSTANT_DIR_NAME, platform_key)
    (components,) = role_targets(pairs, "components")
    (transformer,) = role_targets(pairs, "transformer")
    return components, transformer


def instant_source(platform_key: str) -> InstantSource:
    """Return the Ideogram 4 Instant build source for *platform_key*, read from the ``ideo`` entry of the model metadata."""
    components, transformer = instant_targets(platform_key)
    (transformer_file,) = transformer.allow_patterns
    return InstantSource(
        transformer_repo=transformer.repo_id,
        transformer_revision=transformer.revision,
        transformer_file=transformer_file,
        components=join_huggingface_revision(components.repo_id, components.revision),
        transformer_format=for_platform(model_entry(INSTANT_DIR_NAME)["transformer_format"], platform_key),
    )


def instant_downloads_dir(target: Path) -> Path:
    """Return the hidden folder next to *target* that holds the transformer download while the model is built."""
    return target.with_name(f".{target.name}.download")


def is_instant_dir(path: str | Path) -> bool:
    """Return whether *path* is the Ideogram 4 Instant model folder in the models directory (built or not)."""
    return is_models_dir_child(path, (INSTANT_DIR_NAME,))


def _record(source: InstantSource) -> dict[str, str | int]:
    """Return the build record of a folder built from *source*."""
    return {"version": _RECORD_VERSION, **asdict(source)}


def is_built(model_dir: Path, source: InstantSource) -> bool:
    """Return whether *model_dir* holds a finished build from exactly *source*."""
    return read_build_record(model_dir) == _record(source)


def write_build_record(model_dir: Path, source: InstantSource) -> None:
    """Mark *model_dir* as a finished build from *source*; written last."""
    write_record(model_dir, _record(source))
