"""Plan the Hugging Face downloads of the macOS (mflux, MLX) image, video and enhancer backends without loading them.

Nothing heavy is imported at module level: mflux's weight definitions are imported inside
:func:`image_download_plan`, and the Gemma repo name comes from a lazy lookup.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.model_files import is_ltx_mlx_layout, ltx_mlx_bundled_text_encoder, ltx_mlx_text_encoder_repo
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference

__all__ = ["MLX_LM_ALLOW_PATTERNS", "enhancer_download_plan", "image_download_plan", "video_download_plan"]

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
) -> tuple[DownloadTarget, ...]:
    """Return the downloads of the LTX MLX model *model_ref*: its whole repo at its revision, then the Gemma 3 text encoder repo.

    The vendored loader downloads the model repo without patterns, and ``mlx_lm`` fetches Gemma 3 at the first
    prompt encode, so the plan brings Gemma 3 forward. A model in an LTX version block (LTX-2.5) bundles its Gemma 4
    text encoder in the pack, so the loader never fetches Gemma 3 for it and the plan leaves it out
    (*text_encoder_bundled*). A local folder in the LTX MLX layout needs only Gemma 3, unless it carries its own
    text encoder or *text_encoder_bundled* is set, and then it needs nothing; any other local folder needs
    nothing. The diffusers-only arguments are ignored.

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        family: The model family (``ltx``).
        transformer_source: Ignored (diffusers only).
        latent_upsampler: Ignored (diffusers only).
        upscale: Ignored (diffusers only).
        repo_ignore: Ignored (diffusers only).
        text_encoder_bundled: Whether the model's pack bundles its text encoder, so Gemma 3 is not needed.
    """
    del family, transformer_source, latent_upsampler, upscale, repo_ignore
    gemma = DownloadTarget(ltx_mlx_text_encoder_repo(), allow_patterns=MLX_LM_ALLOW_PATTERNS)
    local = Path(model_ref).expanduser()
    if local.is_dir():
        if not is_ltx_mlx_layout(local) or text_encoder_bundled or ltx_mlx_bundled_text_encoder(local) is not None:
            return ()
        return (gemma,)
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None:
        return ()
    model = DownloadTarget(reference.repo_id, reference.revision)
    return (model,) if text_encoder_bundled else (model, gemma)


def enhancer_download_plan(repo: str, revision: str | None) -> tuple[DownloadTarget, ...]:
    """Return the download of the mlx-lm prompt-enhancer model *repo* (a local folder needs none)."""
    if Path(repo).expanduser().is_dir():
        return ()
    return (DownloadTarget(repo, revision, MLX_LM_ALLOW_PATTERNS),)
