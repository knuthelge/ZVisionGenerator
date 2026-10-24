"""Plan the Hugging Face downloads of the Windows and Linux (diffusers, transformers) image, video and enhancer backends without loading them.

Nothing heavy is imported at module level: no torch, diffusers or transformers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.hub_access import describe_refusal
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.paths import get_ziv_data_dir, parse_huggingface_repo_reference, split_huggingface_revision

if TYPE_CHECKING:
    from zvisiongenerator.utils.h3_preset import H3Preset

__all__ = [
    "DIFFUSERS_IGNORE",
    "DIFFUSERS_ROOT_FILES",
    "SEPARATE_TRANSFORMER_IGNORE",
    "TRANSFORMERS_ALLOW_PATTERNS",
    "enhancer_download_plan",
    "h3_download_plan",
    "image_download_plan",
    "video_download_plan",
    "video_repo_ignore",
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
        RuntimeError: If the Hub refuses the repo (gated, no access or not found); the message says what to do.
        LookupError: If the repo's ``model_index.json`` cannot be read for any other reason.
    """
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None or Path(model_ref).expanduser().is_dir() or family == "ideogram4":
        return ()
    try:
        index = (read_index or _read_hub_model_index)(reference.repo_id, reference.revision)
    except Exception as exc:
        refusal = describe_refusal(exc, reference.repo_id)
        if refusal is not None:
            raise RuntimeError(refusal) from exc
        raise LookupError(f"Could not read model_index.json of '{reference.repo_id}': {exc}") from exc
    allow = ("model_index.json", *DIFFUSERS_ROOT_FILES, *(f"{folder}/*" for folder in _component_folders(index)))
    return (DownloadTarget(reference.repo_id, reference.revision, allow, DIFFUSERS_IGNORE),)


def video_repo_ignore(*, separate_transformer: bool, extra: Sequence[str] = ()) -> tuple[str, ...] | None:
    """Return the patterns an LTX diffusers repo is downloaded without, or ``None`` when it is downloaded whole.

    Args:
        separate_transformer: Whether a single-file transformer replaces the repo's own transformer weights.
        extra: Further patterns to skip, such as the root single-file checkpoints of a model version.
    """
    patterns = (*(SEPARATE_TRANSFORMER_IGNORE if separate_transformer else ()), *extra)
    return patterns or None


def video_download_plan(
    model_ref: str,
    *,
    family: str,
    transformer_source: tuple[str, str] | None = None,
    latent_upsampler: tuple[str, str] | None = None,
    upscale: bool = False,
    repo_ignore: Sequence[str] = (),
    text_encoder_bundled: bool = False,
) -> tuple[DownloadTarget, ...]:
    """Return the downloads of the LTX diffusers model *model_ref*: the repo, the single-file transformer and the latent upsampler.

    The repo keeps the revision of *model_ref* and skips its own transformer weights when *transformer_source*
    replaces them, plus the *repo_ignore* patterns (the same patterns the backend downloads with). The latent
    upsampler is part of the plan only for the two-stage *upscale*, and not when it lives in the model's own repo
    at the same revision. A local folder needs nothing.

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        family: The model family (``ltx``).
        transformer_source: ``(repo_id[@revision], filename)`` of the single-file transformer, or ``None``.
        latent_upsampler: ``(repo_id[@revision], filename)`` of the latent upsampler weights, or ``None``.
        upscale: Whether the run uses the two-stage upscale.
        repo_ignore: Extra patterns the repo download skips.
        text_encoder_bundled: Ignored (the macOS planner uses it to leave Gemma 3 out).
    """
    del family, text_encoder_bundled
    if Path(model_ref).expanduser().is_dir():
        return ()
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None:
        return ()
    repo_target = DownloadTarget(reference.repo_id, reference.revision, ignore_patterns=video_repo_ignore(separate_transformer=bool(transformer_source), extra=repo_ignore))
    targets = [repo_target]
    if transformer_source:
        repo_id, revision = split_huggingface_revision(transformer_source[0])
        targets.append(DownloadTarget(repo_id, revision, allow_patterns=(transformer_source[1],)))
    if upscale and latent_upsampler:
        repo_id, revision = split_huggingface_revision(latent_upsampler[0])
        if (repo_id, revision) != (repo_target.repo_id, repo_target.revision):
            targets.append(DownloadTarget(repo_id, revision, allow_patterns=(latent_upsampler[1],)))
    return tuple(targets)


def _seedvr2_targets() -> tuple[DownloadTarget, ...]:
    """Return the downloads of the SeedVR2 tier a video upscale on this machine would use (none without a SeedVR2 engine)."""
    from zvisiongenerator.backends import accelerator_memory_budget_or_none, get_seedvr2_upscaler
    from zvisiongenerator.downloads.plans import seedvr2_plan
    from zvisiongenerator.seedvr2_loader import build_seedvr2_setup
    from zvisiongenerator.utils.config import load_config

    upscaler = get_seedvr2_upscaler()
    if upscaler is None:
        return ()
    budget = accelerator_memory_budget_or_none()
    models_dir = get_ziv_data_dir() / "models"
    setup = build_seedvr2_setup(
        load_config(),
        weights=upscaler.weights,
        budget_bytes=budget.gpu_bytes if budget else None,
        system_bytes=budget.system_bytes if budget else None,
        models_dir=models_dir,
    )
    return seedvr2_plan(setup.tier.model, models_dir=models_dir, weights=upscaler.weights)


def h3_download_plan(
    model_ref: str,
    preset: H3Preset,
    *,
    upscale: bool = False,
    seedvr2_targets: Callable[[], tuple[DownloadTarget, ...]] | None = None,
) -> tuple[DownloadTarget, ...]:
    """Return the downloads of the MiniMax H3 model *model_ref*: the repository, the text encoder and the turbo LoRA, plus SeedVR2 with *upscale*.

    Each target is the ``(repo_id, revision, allow, ignore)`` the backend's own file resolution downloads with, so a
    target reads as present exactly when the backend finds its files without the network. A reference without a
    revision to the preset's repository takes the preset's revision; a local folder needs no repository download.

    Args:
        model_ref: A local H3 folder or a Hugging Face ``owner/repo[@revision]`` reference.
        preset: The CUDA H3 preset.
        upscale: Whether the run upscales with SeedVR2, which adds the weights of the tier this machine would use.
        seedvr2_targets: Lists the SeedVR2 downloads; defaults to the planner for this machine's tier.
    """
    targets: list[DownloadTarget] = []
    if not Path(model_ref).expanduser().is_dir() and parse_huggingface_repo_reference(model_ref) is not None:
        repo_id, revision = split_huggingface_revision(model_ref)
        if revision is None and repo_id == preset.repo:
            revision = preset.revision
        targets.append(DownloadTarget(repo_id, revision, preset.repo_allow, preset.repo_ignore))
    targets.extend(DownloadTarget(files.repo_id, files.revision, files.files) for files in (preset.text_encoder, preset.lora))
    if upscale:
        targets.extend((seedvr2_targets or _seedvr2_targets)())
    return tuple(targets)


def enhancer_download_plan(repo: str, revision: str | None) -> tuple[DownloadTarget, ...]:
    """Return the download of the transformers prompt-enhancer model *repo* (a local folder needs none)."""
    if Path(repo).expanduser().is_dir():
        return ()
    return (DownloadTarget(repo, revision, TRANSFORMERS_ALLOW_PATTERNS),)
