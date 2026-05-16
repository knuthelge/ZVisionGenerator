"""Video model type detection.

Detects video model family (LTX) from model path or HuggingFace
repo ID using prefix matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from zvisiongenerator.utils.paths import is_explicit_local_path, parse_huggingface_repo_reference


@dataclass(frozen=True)
class VideoModelInfo:
    """Metadata about a detected video model."""

    family: str  # "ltx" | "unknown"
    backend: str  # "ltx" — which VideoBackend to use
    supports_i2v: bool
    default_fps: int
    frame_alignment: int  # LTX: 8 (frames = 8k+1)
    resolution_alignment: int  # LTX: 32


# Prefix → (family, backend, supports_i2v, default_fps, frame_alignment, resolution_alignment)
_VIDEO_MODEL_MAP: dict[str, tuple[str, str, bool, int, int, int]] = {
    "dgrauet/ltx": ("ltx", "ltx", True, 24, 8, 32),
    "dg845/ltx-2.3-diffusers": ("ltx", "ltx", True, 24, 8, 32),
    "lightricks/ltx-video": ("ltx", "ltx", True, 24, 8, 32),
}


def _looks_like_local_model_path(model_path: str) -> bool:
    """Return whether model_path should be treated as a local path.

    Args:
        model_path: Candidate local path or HuggingFace repo ID.

    Returns:
        True when the string uses deterministic local-path syntax.
    """

    stripped = model_path.strip()
    if not stripped:
        return False
    return is_explicit_local_path(stripped)


def detect_video_model(model_path: str) -> VideoModelInfo:
    """Detect video model family from model path or HF repo ID.

    Uses prefix matching since MLX video model repos don't consistently
    use model_index.json with _class_name fields.

    Args:
        model_path: Local path or HuggingFace repo ID.

    Returns:
        VideoModelInfo with detected properties.
    """
    normalized_model_path = model_path.strip()
    normalized_lower = normalized_model_path.lower()

    for prefix, (family, backend, supports_i2v, fps, frame_align, res_align) in _VIDEO_MODEL_MAP.items():
        if normalized_lower.startswith(prefix):
            return VideoModelInfo(
                family=family,
                backend=backend,
                supports_i2v=supports_i2v,
                default_fps=fps,
                frame_alignment=frame_align,
                resolution_alignment=res_align,
            )

    # Configured default repo remains explicitly detectable even when the
    # packaged default is overrideable from user config.
    configured_default = _configured_diffusers_ltx_repo()
    if configured_default and normalized_lower == configured_default.strip().lower():
        return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

    repo_reference = parse_huggingface_repo_reference(normalized_model_path)
    if repo_reference is not None and configured_default and repo_reference.repo_id.lower() == configured_default.strip().lower():
        return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

    if _looks_like_local_model_path(normalized_model_path) and "ltx" in normalized_lower:
        return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

    return VideoModelInfo(
        family="unknown",
        backend="unknown",
        supports_i2v=False,
        default_fps=24,
        frame_alignment=1,
        resolution_alignment=1,
    )


def _configured_diffusers_ltx_repo() -> str | None:
    """Return the configured Windows/Linux diffusers LTX repository, if any."""

    try:
        from zvisiongenerator.utils.config import load_config

        config = load_config()
    except Exception:
        return None

    default_repo = config.get("video_model_presets", {}).get("ltx", {}).get("diffusers", {}).get("default_repo")
    return default_repo.strip() if isinstance(default_repo, str) and default_repo.strip() else None
