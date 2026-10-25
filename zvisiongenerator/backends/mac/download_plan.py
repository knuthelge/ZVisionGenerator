"""Plan the Hugging Face downloads of the macOS (mflux, MLX) image, video and enhancer backends without loading them.

Nothing heavy is imported at module level: mflux's weight definitions are imported inside
:func:`image_download_plan`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.hf_snapshot import pack_for_reference
from zvisiongenerator.utils.model_files import is_ltx_mlx_layout, ltx_mlx_bundled_text_encoder, ltx_mlx_files_bundle_text_encoder
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference

if TYPE_CHECKING:
    from zvisiongenerator.utils.h3_preset import H3MlxPreset, HubFiles

__all__ = ["MLX_LM_ALLOW_PATTERNS", "enhancer_download_plan", "h3_mlx_download_plan", "image_download_plan", "video_download_plan"]

# Mirror of mlx-lm 0.32.0 ``mlx_lm.utils.DEFAULT_ALLOW_PATTERNS``, the files ``mlx_lm.load`` downloads for a repo.
MLX_LM_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)


def _mflux_patterns(family: str) -> list[str] | None:
    """Return the download patterns mflux's own initializer uses for *family*, or ``None`` for a family it does not run."""
    if family == "flux2_klein":
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        return Flux2KleinWeightDefinition.get_download_patterns()
    if family == "zimage":
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

        return ZImageWeightDefinition.get_download_patterns()
    if family == "krea2":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

        return Krea2WeightDefinition.get_download_patterns(ModelConfig.krea2().model_name)
    if family == "ideogram4":
        from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition

        return Ideogram4WeightDefinition.get_download_patterns()
    return None


def image_download_plan(model_ref: str, *, family: str) -> tuple[DownloadTarget, ...]:
    """Return the download of the mflux image model *model_ref*: its repo with the patterns mflux itself fetches.

    A local folder, a reference that is not a repo id, or a family mflux does not run needs no download.

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        family: The detected model family (``flux2_klein``, ``zimage``, ``krea2`` or ``ideogram4``).
    """
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None or Path(model_ref).expanduser().is_dir():
        return ()
    patterns = _mflux_patterns(family)
    if patterns is None:
        return ()
    return (DownloadTarget(reference.repo_id, None, tuple(patterns)),)


def video_download_plan(
    model_ref: str,
    *,
    family: str,
    transformer_source: tuple[str, str] | None = None,
    latent_upsampler: tuple[str, str] | None = None,
    upscale: bool = False,
    repo_ignore: Sequence[str] = (),
    text_encoder_bundled: bool = False,
    mlx_packs: Mapping[str, HubFiles] | None = None,
    mlx_text_encoder: str | None = None,
) -> tuple[DownloadTarget, ...]:
    """Return the downloads of the LTX MLX model *model_ref*: its files at its revision, then the Gemma 3 text encoder.

    A reference to a pinned pack of *mlx_packs* (:func:`~zvisiongenerator.utils.hf_snapshot.pack_for_reference`: its
    repo id in any letter case, with or without its pinned revision) lists exactly the files the app loads from it, at
    its pinned revision, so the plan is the same file set the backend resolves and reads as present exactly when the
    backend finds them without the network. Any other repo reference downloads its whole repo at its own revision, as the
    vendored loader does. Gemma 3 (*mlx_text_encoder*) is planned as well because ``mlx_lm`` would fetch it at the
    first prompt encode, unless the model bundles its Gemma 4 text encoder: a pack does when its file list carries
    one (LTX-2.5), any other model when *text_encoder_bundled* is set, and then the loader never fetches Gemma 3 for it.
    A local folder in the LTX MLX layout needs only Gemma 3, unless it carries its own text encoder or
    *text_encoder_bundled* is set, and then it needs nothing; any other local folder needs nothing. The
    diffusers-only arguments are ignored.

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        family: The model family (``ltx``).
        transformer_source: Ignored (diffusers only).
        latent_upsampler: Ignored (diffusers only).
        upscale: Ignored (diffusers only).
        repo_ignore: Ignored (diffusers only).
        text_encoder_bundled: Whether a model that is no pack of *mlx_packs* bundles its text encoder, so Gemma 3 is not needed.
        mlx_packs: The pinned MLX packs, each with the files the app loads.
        mlx_text_encoder: The Gemma 3 text encoder reference, which a plan that needs Gemma 3 requires.

    Raises:
        ValueError: If the plan needs Gemma 3 and *mlx_text_encoder* is not a Hugging Face repo reference.
    """
    del family, transformer_source, latent_upsampler, upscale, repo_ignore
    local = Path(model_ref).expanduser()
    if local.is_dir():
        if not is_ltx_mlx_layout(local) or text_encoder_bundled or ltx_mlx_bundled_text_encoder(local) is not None:
            return ()
        return (_gemma_target(mlx_text_encoder),)
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None:
        return ()
    pack = pack_for_reference(mlx_packs or {}, model_ref)
    if pack is not None:
        model = DownloadTarget(pack.repo_id, pack.revision, pack.files)
        bundled = ltx_mlx_files_bundle_text_encoder(pack.files)
    else:
        model = DownloadTarget(reference.repo_id, reference.revision)
        bundled = text_encoder_bundled
    return (model,) if bundled else (model, _gemma_target(mlx_text_encoder))


def _gemma_target(text_encoder: str | None) -> DownloadTarget:
    """Return the download of the Gemma 3 text encoder reference *text_encoder*, with the files ``mlx_lm`` loads.

    Raises:
        ValueError: If *text_encoder* is not a Hugging Face repo reference.
    """
    reference = parse_huggingface_repo_reference(text_encoder) if text_encoder is not None else None
    if reference is None:
        raise ValueError(f"The Gemma 3 text encoder must be a Hugging Face repo reference, got {text_encoder!r}.")
    return DownloadTarget(reference.repo_id, reference.revision, MLX_LM_ALLOW_PATTERNS)


def h3_mlx_download_plan(model_ref: str, preset: "H3MlxPreset", lora: "HubFiles") -> tuple[DownloadTarget, ...]:
    """Return the downloads of the MiniMax H3 model on macOS: the INT8 transformer, the ``FL2VA`` components and the turbo LoRA.

    Every target lists its files literally and is the ``(repo_id, revision, allow_patterns)`` the backend's own file
    resolution downloads with, so a target reads as present exactly when the backend finds its files without the
    network. A local H3 folder carries its own components, so it needs no component target, except a converted one,
    which reads its text encoder from the stock download; one that carries a
    transformer needs no transformer target either, because the backend loads that one (or refuses it) and never
    falls back to the preset's.

    Args:
        model_ref: A local H3 folder or a Hugging Face ``owner/repo[@revision]`` reference; a reference takes the
            preset's component files, whatever its revision.
        preset: The macOS preset.
        lora: The turbo LoRA's repository and file.
    """
    from zvisiongenerator.backends.mac.h3_video import local_transformer_dirs
    from zvisiongenerator.utils.h3_converted import read_converted_h3

    local = Path(model_ref).expanduser()
    is_hub = not local.is_dir() and parse_huggingface_repo_reference(model_ref) is not None
    targets = [] if not is_hub and local_transformer_dirs(local) else [DownloadTarget(preset.transformer.repo_id, preset.transformer.revision, preset.transformer.files)]
    # A converted folder reads its text encoder from the stock download, and holds its turbo merge.
    converted = not is_hub and read_converted_h3(local) is not None
    if is_hub or converted:
        targets.append(DownloadTarget(preset.components.repo_id, preset.components.revision, preset.components.files))
    if not converted:
        targets.append(DownloadTarget(lora.repo_id, lora.revision, lora.files))
    return tuple(targets)


def enhancer_download_plan(repo: str, revision: str | None) -> tuple[DownloadTarget, ...]:
    """Return the download of the mlx-lm prompt-enhancer model *repo* (a local folder needs none)."""
    if Path(repo).expanduser().is_dir():
        return ()
    return (DownloadTarget(repo, revision, MLX_LM_ALLOW_PATTERNS),)
