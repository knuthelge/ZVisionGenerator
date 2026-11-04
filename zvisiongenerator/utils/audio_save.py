"""Save a clip as a FLAC file that carries its generation settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.ffmpeg import staging_path
from zvisiongenerator.utils.provenance import config_args
from zvisiongenerator.utils.video_splice import run_ffmpeg, write_pcm16_wav

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__ = ["save_flac"]


def save_flac(samples: NDArray[np.float32], sample_rate: int, path: str | Path, payload: dict[str, Any]) -> Path:
    """Write *samples* to *path* as 16-bit FLAC with *payload* in its ``zvisiongenerator.config`` tag.

    The audio goes through a 16-bit WAV file and ffmpeg's FLAC encoder, at the clip's own rate and channel count. Samples
    beyond -1..1 are clipped by the 16-bit conversion, so bring a clip that can peak higher down first (``limit_peak``).
    Both files are written beside *path* under a hidden name, and *path* appears only once it is complete.

    Args:
        samples: Float32 samples shaped ``(channels, frames)``, in -1..1.
        sample_rate: Samples per second.
        path: The ``.flac`` file to write; an existing file is replaced.
        payload: The config payload to embed.

    Returns:
        *path*.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    path = Path(path)
    wav = staging_path(path.with_suffix(".wav"))
    staged = staging_path(path)
    try:
        write_pcm16_wav(samples, wav, sample_rate)
        run_ffmpeg(
            ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-map_metadata", "-1", "-c:a", "flac", "-sample_fmt", "s16", *config_args(payload), str(staged)],
            "ffmpeg could not write the audio file",
        )
        staged.replace(path)
    finally:
        wav.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
    return path
