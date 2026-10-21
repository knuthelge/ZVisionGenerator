"""Plan the Hugging Face downloads of the Windows and Linux (diffusers, transformers) image, video and enhancer backends without loading them.

Nothing heavy is imported at module level: no torch, diffusers or transformers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference

__all__ = [
    "DIFFUSERS_IGNORE",
    "DIFFUSERS_ROOT_FILES",
    "SEPARATE_TRANSFORMER_IGNORE",
    "TRANSFORMERS_ALLOW_PATTERNS",
    "enhancer_download_plan",
    "image_download_plan",
    "video_download_plan",
]

# An LTX diffusers repo is downloaded without its own transformer weights when a single-file transformer replaces them.
SEPARATE_TRANSFORMER_IGNORE: tuple[str, ...] = ("transformer/*.safetensors", "transformer/*.safetensors.index.json")

# Root files ``DiffusionPipeline.download`` keeps (diffusers 0.41 ``pipeline_utils.py``: the scheduler, config and model
# index names plus ``TRANSFORMERS_COMPONENT_AUX_FILES``).
DIFFUSERS_ROOT_FILES: tuple[str, ...] = (
    "config.json",
    "scheduler_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_PRECISION_VARIANTS = ("fp16", "bf16", "fp32", "fp8")
# Weight formats and precision variants the diffusers loaders do not use (they load plain safetensors).
DIFFUSERS_IGNORE: tuple[str, ...] = (
    "*.bin",
    "*.pth",
    "*.ckpt",
    "*.onnx",
    "*.msgpack",
    "*.h5",
    *(pattern for variant in _PRECISION_VARIANTS for pattern in (f"*.{variant}.safetensors", f"*.{variant}-*.safetensors", f"*.{variant}.safetensors.index.json")),
)

# Files ``AutoTokenizer`` and ``AutoModelForCausalLM`` read from a prompt-enhancer repo.
TRANSFORMERS_ALLOW_PATTERNS: tuple[str, ...] = ("*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*")


def _read_hub_model_index(repo_id: str, revision: str | None) -> dict[str, Any]:
    """Return the parsed ``model_index.json`` of a Hub repo, from the local cache when present, else from the network."""
    from huggingface_hub import hf_hub_download

    from zvisiongenerator.utils.image_model_detect import _hf_hub_file

    path = _hf_hub_file(hf_hub_download, repo_id, "model_index.json", revision)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _component_folders(index: dict[str, Any]) -> list[str]:
    """Return the component folders of a diffusers ``model_index.json`` (entries naming a library and a class)."""
    return [name for name, value in index.items() if not name.startswith("_") and isinstance(value, list | tuple) and value and isinstance(value[0], str)]


def image_download_plan(
    model_ref: str,
    *,
    family: str,
    read_index: Callable[[str, str | None], dict[str, Any]] | None = None,
) -> tuple[DownloadTarget, ...]:
    """Return the download of the diffusers image model *model_ref*: the component folders its ``model_index.json`` names.

    The patterns follow what ``DiffusionPipeline.download`` fetches, so single-file checkpoints at the repo root
    (Klein's ``flux-2-klein-9b.safetensors``, Krea's ``turbo.safetensors``) and precision variants are left out.
    A local folder, a reference that is not a repo id, and an Ideogram 4 base repo (refused on this platform) need no download.

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        family: The detected model family.
        read_index: Reads a repo's ``model_index.json`` as ``(repo_id, revision) -> dict``; defaults to the Hub (local cache first).

    Raises:
        LookupError: If the repo's ``model_index.json`` cannot be read.
    """
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None or Path(model_ref).expanduser().is_dir() or family == "ideogram4":
        return ()
    try:
        index = (read_index or _read_hub_model_index)(reference.repo_id, reference.revision)
    except Exception as exc:
        raise LookupError(f"Could not read model_index.json of '{reference.repo_id}': {exc}") from exc
    allow = ("model_index.json", *DIFFUSERS_ROOT_FILES, *(f"{folder}/*" for folder in _component_folders(index)))
    return (DownloadTarget(reference.repo_id, reference.revision, allow, DIFFUSERS_IGNORE),)


def video_download_plan(
    model_ref: str,
    *,
    family: str,
    transformer_source: tuple[str, str] | None = None,
    latent_upsampler: tuple[str, str] | None = None,
    upscale: bool = False,
) -> tuple[DownloadTarget, ...]:
    """Return the downloads of the LTX diffusers model *model_ref*: the repo, the single-file transformer and the latent upsampler.

    The repo skips its own transformer weights when *transformer_source* replaces them (the same patterns the
    backend downloads with). The latent upsampler is part of the plan only for the two-stage *upscale*. A local folder needs nothing.

    Args:
        model_ref: A local path or a Hugging Face repo id.
        family: The model family (``ltx``).
        transformer_source: ``(repo_id, filename)`` of the single-file transformer, or ``None``.
        latent_upsampler: ``(repo_id, filename)`` of the latent upsampler weights, or ``None``.
        upscale: Whether the run uses the two-stage upscale.
    """
    del family
    if Path(model_ref).expanduser().is_dir():
        return ()
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None:
        return ()
    targets = [DownloadTarget(reference.repo_id, ignore_patterns=SEPARATE_TRANSFORMER_IGNORE if transformer_source else None)]
    if transformer_source:
        targets.append(DownloadTarget(transformer_source[0], allow_patterns=(transformer_source[1],)))
    if upscale and latent_upsampler:
        targets.append(DownloadTarget(latent_upsampler[0], allow_patterns=(latent_upsampler[1],)))
    return tuple(targets)


def enhancer_download_plan(repo: str, revision: str | None) -> tuple[DownloadTarget, ...]:
    """Return the download of the transformers prompt-enhancer model *repo* (a local folder needs none)."""
    if Path(repo).expanduser().is_dir():
        return ()
    return (DownloadTarget(repo, revision, TRANSFORMERS_ALLOW_PATTERNS),)
