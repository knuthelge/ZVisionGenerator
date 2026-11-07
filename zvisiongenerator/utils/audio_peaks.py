"""Waveform peaks of an audio file: 100 small numbers the gallery draws as a waveform without loading the file."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from zvisiongenerator.utils.ffmpeg import tool_error

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["PEAK_COUNT", "PEAK_MAX", "PEAK_SAMPLE_RATE", "read_waveform", "read_waveform_peaks", "waveform_peaks"]

PEAK_COUNT: Final = 100
"""How many peaks a file's waveform has."""

PEAK_MAX: Final = 255
"""The largest peak value: the loudest stretch of the file."""

PEAK_SAMPLE_RATE: Final = 8000
"""The rate a file is decoded at to find its peaks; enough for a drawing, and a minute of audio is under 2 MB."""


def waveform_peaks(samples: NDArray[np.float32], count: int = PEAK_COUNT) -> list[int]:
    """Return *count* peaks of *samples*, each 0 to 255, for drawing the waveform.

    The audio is cut into *count* equal stretches. A stretch's peak is its largest absolute sample over all channels,
    and the values are scaled so the loudest stretch is 255 and silence is 0, so a quiet take still shows its shape.

    Args:
        samples: Float32 samples shaped ``(channels, frames)`` or ``(frames,)``.
        count: How many peaks to return.

    Returns:
        *count* integers; all 0 for silent or empty audio.

    Raises:
        ValueError: If *count* is below 1.
    """
    if count < 1:
        raise ValueError("waveform_peaks needs a count of at least 1.")
    magnitude = np.abs(np.asarray(samples, dtype=np.float32))
    if magnitude.ndim > 1:
        magnitude = magnitude.max(axis=0)
    frames = magnitude.shape[0]
    if frames == 0:
        return [0] * count
    edges = np.linspace(0, frames, count + 1).astype(np.int64)
    # A stretch is at least one frame, so audio shorter than *count* frames repeats its frames instead of reading as silence.
    peaks = np.array([magnitude[edge : max(end, edge + 1)].max() for edge, end in zip(edges[:-1], edges[1:], strict=True)], dtype=np.float64)
    loudest = float(peaks.max())
    if loudest <= 0:
        return [0] * count
    return [int(value) for value in np.rint(peaks / loudest * PEAK_MAX)]


def read_waveform(path: str | Path, count: int = PEAK_COUNT, *, timeout: float | None = None) -> tuple[list[int], float]:
    """Decode the first audio stream of *path* to mono at 8 kHz with one ffmpeg call and return its :func:`waveform_peaks` and its length in seconds.

    The length is the decoded sample count over ``PEAK_SAMPLE_RATE``, so it costs nothing beyond the decode.

    Args:
        path: The audio file.
        count: How many peaks to return.
        timeout: Seconds ffmpeg may run before it is stopped; ``None`` waits as long as it takes.

    Raises:
        RuntimeError: If ffmpeg cannot decode the file or runs past *timeout*.
    """
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(PEAK_SAMPLE_RATE), "-f", "f32le", "-"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg took longer than {timeout:g} s to decode {path}") from exc
    if result.returncode != 0:
        raise tool_error(f"ffmpeg could not decode {path}", result.stderr)
    data = result.stdout
    samples = np.frombuffer(data[: len(data) - len(data) % 4], dtype=np.float32)
    return waveform_peaks(samples, count), samples.shape[0] / PEAK_SAMPLE_RATE


def read_waveform_peaks(path: str | Path, count: int = PEAK_COUNT) -> list[int]:
    """Decode the first audio stream of *path* to mono at 8 kHz with one ffmpeg call and return its :func:`waveform_peaks`.

    Raises:
        RuntimeError: If ffmpeg cannot decode the file.
    """
    return read_waveform(path, count)[0]
