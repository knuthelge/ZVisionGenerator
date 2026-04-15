"""Video model type detection.

Detects video model family (LTX) from model path or HuggingFace
repo ID using prefix matching.
"""

from __future__ import annotations

from dataclasses import dataclass


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
}


def detect_video_model(model_path: str) -> VideoModelInfo:
    """Detect video model family from model path or HF repo ID.

    Uses prefix matching since MLX video model repos don't consistently
    use model_index.json with _class_name fields.

    Args:
        model_path: Local path or HuggingFace repo ID.

    Returns:
        VideoModelInfo with detected properties.
    """
    for prefix, (family, backend, supports_i2v, fps, frame_align, res_align) in _VIDEO_MODEL_MAP.items():
        if model_path.startswith(prefix):
            return VideoModelInfo(
                family=family,
                backend=backend,
                supports_i2v=supports_i2v,
                default_fps=fps,
                frame_alignment=frame_align,
                resolution_alignment=res_align,
            )

    # Fallback: detect from substrings in the path / basename (local paths)
    path_lower = model_path.lower()
    if "ltx" in path_lower:
        return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

    return VideoModelInfo(
        family="unknown",
        backend="unknown",
        supports_i2v=False,
        default_fps=24,
        frame_alignment=1,
        resolution_alignment=1,
    )
